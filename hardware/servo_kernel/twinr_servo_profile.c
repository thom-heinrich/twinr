/*
 * twinr_servo_profile.c - bounded motion probe for the Twinr servo kernel module
 *
 * This small C utility drives /sys/class/twinr_servo/servo0 with an eased
 * point-to-point move. It exists to keep motion-profile experimentation out of
 * the kernel module itself while still avoiding Python or high-level runtime
 * dependencies during Pi acceptance work.
 *
 * The profile supports:
 * - a minimum-jerk trajectory between start and target pulse widths
 * - micro-step gating so tiny updates are skipped
 * - an optional short breakaway pulse to overcome deadband/stiction at motion start
 * - persisted last-target tracking so the next bounded move starts from the
 *   last commanded end position instead of blindly snapping to a stale start
 * - bounded settle time followed by clean disable + GPIO release
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_SYSFS_ROOT "/sys/class/twinr_servo/servo0"
#define DEFAULT_GPIO 18
#define DEFAULT_PERIOD_US 20000
#define DEFAULT_SETTLE_MS 1000
#define DEFAULT_CADENCE_MS 52
#define DEFAULT_GATE_US 6
#define DEFAULT_BREAKAWAY_US 18
#define DEFAULT_BREAKAWAY_HOLD_MS 140
#define DEFAULT_STATE_FILE "/var/tmp/twinr_servo_profile_last_us"

struct motion_config {
	const char *sysfs_root;
	const char *state_file;
	int gpio;
	int period_us;
	int start_us;
	int target_us;
	int duration_ms;
	int settle_ms;
	int cadence_ms;
	int gate_us;
	int breakaway_us;
	int breakaway_hold_ms;
	bool override_state;
	bool start_explicit;
	bool disable_state_tracking;
	bool verbose;
};

struct cleanup_state {
	const char *sysfs_root;
	bool armed;
};

static struct cleanup_state g_cleanup = {0};

static void usage(FILE *stream, const char *argv0)
{
	fprintf(
		stream,
		"Usage: %s --start-us N --target-us N --duration-ms N [options]\n"
		"\n"
		"Options:\n"
		"  --sysfs-root PATH          Sysfs device root (default %s)\n"
		"  --gpio N                   GPIO line to claim (default %d)\n"
		"  --period-us N              PWM period in microseconds (default %d)\n"
		"  --state-file PATH          Persisted last-target file (default %s)\n"
		"  --no-state                 Disable persisted last-target safety checks\n"
		"  --override-state           Allow --start-us to differ from the persisted state\n"
		"  --settle-ms N              Hold at target before release (default %d)\n"
		"  --cadence-ms N             Control cadence in milliseconds (default %d)\n"
		"  --gate-us N                Skip updates below this delta (default %d)\n"
		"  --breakaway-us N           Initial kick toward target in microseconds (default %d)\n"
		"  --breakaway-hold-ms N      Hold the kick before easing (default %d)\n"
		"  --verbose                  Print each committed pulse write\n",
		argv0,
		DEFAULT_SYSFS_ROOT,
		DEFAULT_GPIO,
		DEFAULT_PERIOD_US,
		DEFAULT_STATE_FILE,
		DEFAULT_SETTLE_MS,
		DEFAULT_CADENCE_MS,
		DEFAULT_GATE_US,
		DEFAULT_BREAKAWAY_US,
		DEFAULT_BREAKAWAY_HOLD_MS);
}

static int write_attr(const char *root, const char *name, int value)
{
	char path[PATH_MAX];
	FILE *file;

	if (snprintf(path, sizeof(path), "%s/%s", root, name) >= (int)sizeof(path)) {
		fprintf(stderr, "path too long for %s\n", name);
		return -ENAMETOOLONG;
	}

	file = fopen(path, "w");
	if (file == NULL) {
		perror(path);
		return -errno;
	}
	if (fprintf(file, "%d", value) < 0) {
		perror(path);
		fclose(file);
		return -EIO;
	}
	if (fclose(file) != 0) {
		perror(path);
		return -errno;
	}
	return 0;
}

static void sleep_ms(int milliseconds)
{
	struct timespec request;

	if (milliseconds <= 0)
		return;
	request.tv_sec = milliseconds / 1000;
	request.tv_nsec = (long)(milliseconds % 1000) * 1000000L;
	while (nanosleep(&request, &request) != 0) {
		if (errno != EINTR)
			break;
	}
}

static void cleanup_servo(void)
{
	if (!g_cleanup.armed || g_cleanup.sysfs_root == NULL)
		return;
	(void)write_attr(g_cleanup.sysfs_root, "enabled", 0);
	(void)write_attr(g_cleanup.sysfs_root, "gpio", -1);
	g_cleanup.armed = false;
}

static void handle_signal(int signo)
{
	(void)signo;
	cleanup_servo();
	_exit(128 + signo);
}

static double min_jerk(double t)
{
	double t2 = t * t;
	double t3 = t2 * t;
	double t4 = t3 * t;
	double t5 = t4 * t;

	return 10.0 * t3 - 15.0 * t4 + 6.0 * t5;
}

static int rounded_lerp(int start, int target, double weight)
{
	double value = (double)start + ((double)target - (double)start) * weight;

	return (int)(value >= 0.0 ? value + 0.5 : value - 0.5);
}

static int abs_int(int value)
{
	return value < 0 ? -value : value;
}

static int parse_int(const char *text, const char *label)
{
	char *end = NULL;
	long value;

	errno = 0;
	value = strtol(text, &end, 10);
	if (errno != 0 || end == text || *end != '\0' || value < INT_MIN || value > INT_MAX) {
		fprintf(stderr, "invalid %s: %s\n", label, text);
		exit(2);
	}
	return (int)value;
}

static int load_state_file(const char *path, int *pulse_us)
{
	FILE *file;
	char line[64];
	char *end = NULL;
	long value;

	file = fopen(path, "r");
	if (file == NULL) {
		if (errno == ENOENT)
			return 1;
		perror(path);
		return -errno;
	}
	if (fgets(line, sizeof(line), file) == NULL) {
		if (fclose(file) != 0)
			perror(path);
		fprintf(stderr, "failed to read %s\n", path);
		return -EIO;
	}
	if (fclose(file) != 0) {
		perror(path);
		return -errno;
	}

	errno = 0;
	value = strtol(line, &end, 10);
	if (errno != 0 || end == line || (*end != '\0' && *end != '\n') || value < INT_MIN || value > INT_MAX) {
		fprintf(stderr, "invalid persisted state in %s: %s\n", path, line);
		return -EINVAL;
	}
	*pulse_us = (int)value;
	return 0;
}

static int save_state_file(const char *path, int pulse_us)
{
	FILE *file;

	file = fopen(path, "w");
	if (file == NULL) {
		perror(path);
		return -errno;
	}
	if (fprintf(file, "%d\n", pulse_us) < 0) {
		perror(path);
		fclose(file);
		return -EIO;
	}
	if (fclose(file) != 0) {
		perror(path);
		return -errno;
	}
	return 0;
}

static void parse_args(int argc, char **argv, struct motion_config *config)
{
	static const struct option options[] = {
		{"sysfs-root", required_argument, NULL, 'r'},
		{"gpio", required_argument, NULL, 'g'},
		{"period-us", required_argument, NULL, 'p'},
		{"state-file", required_argument, NULL, 'f'},
		{"no-state", no_argument, NULL, 'n'},
		{"override-state", no_argument, NULL, 'o'},
		{"start-us", required_argument, NULL, 's'},
		{"target-us", required_argument, NULL, 't'},
		{"duration-ms", required_argument, NULL, 'd'},
		{"settle-ms", required_argument, NULL, 'S'},
		{"cadence-ms", required_argument, NULL, 'c'},
		{"gate-us", required_argument, NULL, 'G'},
		{"breakaway-us", required_argument, NULL, 'b'},
		{"breakaway-hold-ms", required_argument, NULL, 'B'},
		{"verbose", no_argument, NULL, 'v'},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0},
	};
	bool start_seen = false;
	bool target_seen = false;
	bool duration_seen = false;
	int opt;

		*config = (struct motion_config){
		.sysfs_root = DEFAULT_SYSFS_ROOT,
		.state_file = DEFAULT_STATE_FILE,
		.gpio = DEFAULT_GPIO,
		.period_us = DEFAULT_PERIOD_US,
		.settle_ms = DEFAULT_SETTLE_MS,
		.cadence_ms = DEFAULT_CADENCE_MS,
		.gate_us = DEFAULT_GATE_US,
		.breakaway_us = DEFAULT_BREAKAWAY_US,
		.breakaway_hold_ms = DEFAULT_BREAKAWAY_HOLD_MS,
	};

	while ((opt = getopt_long(argc, argv, "r:g:p:f:nos:t:d:S:c:G:b:B:vh", options, NULL)) != -1) {
		switch (opt) {
		case 'r':
			config->sysfs_root = optarg;
			break;
		case 'g':
			config->gpio = parse_int(optarg, "gpio");
			break;
		case 'p':
			config->period_us = parse_int(optarg, "period-us");
			break;
		case 'f':
			config->state_file = optarg;
			break;
		case 'n':
			config->disable_state_tracking = true;
			break;
		case 'o':
			config->override_state = true;
			break;
		case 's':
			config->start_us = parse_int(optarg, "start-us");
			start_seen = true;
			config->start_explicit = true;
			break;
		case 't':
			config->target_us = parse_int(optarg, "target-us");
			target_seen = true;
			break;
		case 'd':
			config->duration_ms = parse_int(optarg, "duration-ms");
			duration_seen = true;
			break;
		case 'S':
			config->settle_ms = parse_int(optarg, "settle-ms");
			break;
		case 'c':
			config->cadence_ms = parse_int(optarg, "cadence-ms");
			break;
		case 'G':
			config->gate_us = parse_int(optarg, "gate-us");
			break;
		case 'b':
			config->breakaway_us = parse_int(optarg, "breakaway-us");
			break;
		case 'B':
			config->breakaway_hold_ms = parse_int(optarg, "breakaway-hold-ms");
			break;
		case 'v':
			config->verbose = true;
			break;
		case 'h':
			usage(stdout, argv[0]);
			exit(0);
		default:
			usage(stderr, argv[0]);
			exit(2);
		}
	}

	if (!target_seen || !duration_seen) {
		usage(stderr, argv[0]);
		exit(2);
	}
	if (config->duration_ms <= 0 || config->cadence_ms <= 0 || config->gate_us < 0) {
		fprintf(stderr, "duration/cadence must be > 0 and gate must be >= 0\n");
		exit(2);
	}
	if (!config->disable_state_tracking) {
		int persisted_start_us = 0;
		int state_rc = load_state_file(config->state_file, &persisted_start_us);

		if (state_rc < 0)
			exit(2);
		if (state_rc == 0) {
			if (start_seen) {
				if (config->start_us != persisted_start_us && !config->override_state) {
					fprintf(
						stderr,
						"start-us %d disagrees with persisted state %d in %s; "
						"use --override-state to force a reset\n",
						config->start_us,
						persisted_start_us,
						config->state_file);
					exit(2);
				}
			} else {
				config->start_us = persisted_start_us;
				start_seen = true;
			}
		}
	}
	if (!start_seen) {
		fprintf(stderr, "start-us is required when no persisted state is available\n");
		exit(2);
	}
}

static int maybe_write_pulse(
	const struct motion_config *config,
	int pulse_us,
	int *last_written_us,
	bool force)
{
	if (!force && abs_int(pulse_us - *last_written_us) < config->gate_us)
		return 0;
	if (config->verbose)
		printf("pulse_width_us=%d\n", pulse_us);
	if (write_attr(config->sysfs_root, "pulse_width_us", pulse_us) != 0)
		return -1;
	*last_written_us = pulse_us;
	return 0;
}

static int run_profile(const struct motion_config *config)
{
	int direction = 0;
	int last_written_us = config->start_us;
	int profile_start_us = config->start_us;
	int steps = config->duration_ms / config->cadence_ms;
	int breakaway_target;
	int step;

	if (write_attr(config->sysfs_root, "gpio", config->gpio) != 0)
		return 1;
	if (write_attr(config->sysfs_root, "period_us", config->period_us) != 0)
		return 1;
	if (write_attr(config->sysfs_root, "pulse_width_us", config->start_us) != 0)
		return 1;
	if (write_attr(config->sysfs_root, "enabled", 1) != 0)
		return 1;
	g_cleanup.sysfs_root = config->sysfs_root;
	g_cleanup.armed = true;

	sleep_ms(800);

	if (config->target_us > config->start_us)
		direction = 1;
	else if (config->target_us < config->start_us)
		direction = -1;

	if (direction != 0 && config->breakaway_us > 0 && config->breakaway_hold_ms > 0) {
		int remaining_us = abs_int(config->target_us - config->start_us);
		int kick_us = config->breakaway_us < remaining_us ? config->breakaway_us : remaining_us;

		breakaway_target = config->start_us + direction * kick_us;
		if (maybe_write_pulse(config, breakaway_target, &last_written_us, false) != 0)
			return 1;
		profile_start_us = breakaway_target;
		sleep_ms(config->breakaway_hold_ms);
	}

	if (steps < 1)
		steps = 1;

	for (step = 0; step <= steps; ++step) {
		double t = (double)step / (double)steps;
		int pulse_us = rounded_lerp(profile_start_us, config->target_us, min_jerk(t));

		if (maybe_write_pulse(config, pulse_us, &last_written_us, step == steps) != 0)
			return 1;
		if (step < steps)
			sleep_ms(config->cadence_ms);
	}

	sleep_ms(config->settle_ms);
	if (!config->disable_state_tracking && save_state_file(config->state_file, config->target_us) != 0)
		return 1;
	return 0;
}

int main(int argc, char **argv)
{
	struct motion_config config;
	struct sigaction action;
	int rc;

	parse_args(argc, argv, &config);

	memset(&action, 0, sizeof(action));
	action.sa_handler = handle_signal;
	sigaction(SIGINT, &action, NULL);
	sigaction(SIGTERM, &action, NULL);

	rc = run_profile(&config);
	cleanup_servo();
	return rc;
}
