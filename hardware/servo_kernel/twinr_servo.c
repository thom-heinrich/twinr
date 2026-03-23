/*
 * twinr_servo.c - Single-servo kernel driver for calm Twinr body motion
 *
 * This out-of-tree module provides one hrtimer-driven PWM output for a
 * standard hobby servo. It exists because userspace PWM helpers on the Pi did
 * not produce senior-safe movement quality for Twinr's body-follow axis.
 *
 * The module exposes one sysfs-controlled singleton device at:
 *
 *   /sys/class/twinr_servo/servo0/
 *
 * Attributes:
 *   gpio            - GPIO line number, -1 when unconfigured
 *   period_us       - Servo period in microseconds, default 20000
 *   pulse_width_us  - High pulse width in microseconds, default 1500
 *   enabled         - 0 or 1
 *
 * Change ``pulse_width_us`` while enabled to retarget the servo. Disable the
 * device before reassigning ``gpio``.
 */

#include <linux/device.h>
#include <linux/err.h>
#include <linux/gpio/consumer.h>
#include <linux/gpio/driver.h>
#include <linux/gpio/machine.h>
#include <linux/hrtimer.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/kstrtox.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>

#define TWINR_SERVO_LABEL "twinr_servo"
#define TWINR_SERVO_DEVICE_NAME "servo0"
#define TWINR_SERVO_GPIOCHIP_LABEL "pinctrl-rp1"
#define TWINR_SERVO_DEFAULT_PERIOD_US 20000U
#define TWINR_SERVO_DEFAULT_PULSE_WIDTH_US 1500U
#define TWINR_SERVO_MIN_PERIOD_US 5000U
#define TWINR_SERVO_MAX_PERIOD_US 50000U
#define TWINR_SERVO_MIN_PULSE_WIDTH_US 500U
#define TWINR_SERVO_MAX_PULSE_WIDTH_US 2500U

struct twinr_servo_device {
	struct class *class;
	struct device *dev;
	struct gpio_desc *desc;
	struct hrtimer timer;
	struct mutex config_lock;
	spinlock_t state_lock;
	int gpio;
	bool enabled;
	bool phase_high;
	u32 period_us;
	u32 pulse_width_us;
};

static struct twinr_servo_device twinr_servo = {
	.gpio = -1,
	.period_us = TWINR_SERVO_DEFAULT_PERIOD_US,
	.pulse_width_us = TWINR_SERVO_DEFAULT_PULSE_WIDTH_US,
};

static void twinr_servo_drive_low_locked(struct twinr_servo_device *servo)
{
	if (servo->desc != NULL)
		gpiod_set_value(servo->desc, 0);
}

static void twinr_servo_stop_locked(struct twinr_servo_device *servo)
{
	unsigned long flags;

	spin_lock_irqsave(&servo->state_lock, flags);
	servo->enabled = false;
	servo->phase_high = false;
	spin_unlock_irqrestore(&servo->state_lock, flags);

	hrtimer_cancel(&servo->timer);
	twinr_servo_drive_low_locked(servo);
}

static int twinr_servo_request_gpio_locked(struct twinr_servo_device *servo, int gpio)
{
	struct gpio_chip *gpio_chip;
	struct gpio_desc *desc;
	struct gpio_device *gpio_device;
	int ret;

	if (gpio < 0) {
		if (servo->desc != NULL) {
			twinr_servo_drive_low_locked(servo);
			gpiochip_free_own_desc(servo->desc);
			servo->desc = NULL;
		}
		servo->gpio = -1;
		return 0;
	}
	if (servo->enabled)
		return -EBUSY;
	if (servo->desc != NULL && servo->gpio == gpio)
		return 0;
	if (servo->desc != NULL) {
		twinr_servo_drive_low_locked(servo);
		gpiochip_free_own_desc(servo->desc);
		servo->desc = NULL;
		servo->gpio = -1;
	}
	gpio_device = gpio_device_find_by_label(TWINR_SERVO_GPIOCHIP_LABEL);
	if (gpio_device == NULL)
		return -ENODEV;
	gpio_chip = gpio_device_get_chip(gpio_device);
	if (gpio_chip == NULL)
		return -ENODEV;
	if (gpio < 0 || gpio >= gpio_chip->ngpio)
		return -EINVAL;
	desc = gpiochip_request_own_desc(
		gpio_chip,
		gpio,
		TWINR_SERVO_LABEL,
		GPIO_LOOKUP_FLAGS_DEFAULT,
		GPIOD_OUT_LOW);
	if (IS_ERR(desc)) {
		ret = PTR_ERR(desc);
		pr_warn("twinr_servo: failed to claim %s GPIO%d: %d\n",
			TWINR_SERVO_GPIOCHIP_LABEL,
			gpio,
			ret);
		return ret;
	}
	servo->desc = desc;
	servo->gpio = gpio;
	pr_info("twinr_servo: claimed %s GPIO%d\n", TWINR_SERVO_GPIOCHIP_LABEL, gpio);
	return 0;
}

static enum hrtimer_restart twinr_servo_timer_callback(struct hrtimer *timer)
{
	struct twinr_servo_device *servo = container_of(timer, struct twinr_servo_device, timer);
	unsigned long flags;
	bool enabled;
	bool phase_high;
	u32 period_us;
	u32 pulse_width_us;
	ktime_t interval;

	spin_lock_irqsave(&servo->state_lock, flags);
	enabled = servo->enabled && servo->desc != NULL && servo->gpio >= 0;
	if (!enabled) {
		servo->phase_high = false;
		spin_unlock_irqrestore(&servo->state_lock, flags);
		return HRTIMER_NORESTART;
	}
	phase_high = !servo->phase_high;
	servo->phase_high = phase_high;
	period_us = servo->period_us;
	pulse_width_us = min(servo->pulse_width_us, servo->period_us);
	if (pulse_width_us == 0)
		pulse_width_us = 1;
	if (pulse_width_us >= period_us)
		pulse_width_us = period_us - 1;
	spin_unlock_irqrestore(&servo->state_lock, flags);

	gpiod_set_value(servo->desc, phase_high ? 1 : 0);
	interval = ktime_set(0, (phase_high ? pulse_width_us : (period_us - pulse_width_us)) * NSEC_PER_USEC);
	hrtimer_forward_now(timer, interval);
	return HRTIMER_RESTART;
}

static void twinr_servo_start_locked(struct twinr_servo_device *servo)
{
	unsigned long flags;

	spin_lock_irqsave(&servo->state_lock, flags);
	servo->phase_high = false;
	servo->enabled = true;
	spin_unlock_irqrestore(&servo->state_lock, flags);

	hrtimer_start(&servo->timer, ktime_set(0, 1), HRTIMER_MODE_REL_PINNED);
}

static ssize_t gpio_show(struct device *dev, struct device_attribute *attr, char *buf)
{
	struct twinr_servo_device *servo = dev_get_drvdata(dev);

	return sysfs_emit(buf, "%d\n", servo->gpio);
}

static ssize_t gpio_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count)
{
	struct twinr_servo_device *servo = dev_get_drvdata(dev);
	long value;
	int ret;

	ret = kstrtol(buf, 0, &value);
	if (ret)
		return ret;

	mutex_lock(&servo->config_lock);
	ret = twinr_servo_request_gpio_locked(servo, (int)value);
	mutex_unlock(&servo->config_lock);

	return ret ? ret : (ssize_t)count;
}

static ssize_t period_us_show(struct device *dev, struct device_attribute *attr, char *buf)
{
	struct twinr_servo_device *servo = dev_get_drvdata(dev);

	return sysfs_emit(buf, "%u\n", servo->period_us);
}

static ssize_t period_us_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count)
{
	struct twinr_servo_device *servo = dev_get_drvdata(dev);
	unsigned int value;
	int ret;

	ret = kstrtouint(buf, 0, &value);
	if (ret)
		return ret;
	if (value < TWINR_SERVO_MIN_PERIOD_US || value > TWINR_SERVO_MAX_PERIOD_US)
		return -ERANGE;

	mutex_lock(&servo->config_lock);
	if (servo->enabled) {
		mutex_unlock(&servo->config_lock);
		return -EBUSY;
	}
	servo->period_us = value;
	if (servo->pulse_width_us >= servo->period_us)
		servo->pulse_width_us = servo->period_us - 1;
	mutex_unlock(&servo->config_lock);

	return count;
}

static ssize_t pulse_width_us_show(struct device *dev, struct device_attribute *attr, char *buf)
{
	struct twinr_servo_device *servo = dev_get_drvdata(dev);

	return sysfs_emit(buf, "%u\n", servo->pulse_width_us);
}

static ssize_t pulse_width_us_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count)
{
	struct twinr_servo_device *servo = dev_get_drvdata(dev);
	unsigned int value;
	unsigned long flags;
	int ret;

	ret = kstrtouint(buf, 0, &value);
	if (ret)
		return ret;
	if (value < TWINR_SERVO_MIN_PULSE_WIDTH_US || value > TWINR_SERVO_MAX_PULSE_WIDTH_US)
		return -ERANGE;

	mutex_lock(&servo->config_lock);
	if (value >= servo->period_us) {
		mutex_unlock(&servo->config_lock);
		return -ERANGE;
	}
	spin_lock_irqsave(&servo->state_lock, flags);
	servo->pulse_width_us = value;
	spin_unlock_irqrestore(&servo->state_lock, flags);
	mutex_unlock(&servo->config_lock);

	return count;
}

static ssize_t enabled_show(struct device *dev, struct device_attribute *attr, char *buf)
{
	struct twinr_servo_device *servo = dev_get_drvdata(dev);

	return sysfs_emit(buf, "%u\n", servo->enabled ? 1 : 0);
}

static ssize_t enabled_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count)
{
	struct twinr_servo_device *servo = dev_get_drvdata(dev);
	bool enable;
	int ret;

	ret = kstrtobool(buf, &enable);
	if (ret)
		return ret;

	mutex_lock(&servo->config_lock);
	if (enable) {
		if (servo->desc == NULL || servo->gpio < 0) {
			ret = -ENODEV;
			goto out;
		}
		if (!servo->enabled)
			twinr_servo_start_locked(servo);
	} else if (servo->enabled) {
		twinr_servo_stop_locked(servo);
	}
	ret = 0;

out:
	mutex_unlock(&servo->config_lock);
	return ret ? ret : (ssize_t)count;
}

static DEVICE_ATTR_RW(gpio);
static DEVICE_ATTR_RW(period_us);
static DEVICE_ATTR_RW(pulse_width_us);
static DEVICE_ATTR_RW(enabled);

static struct attribute *twinr_servo_attrs[] = {
	&dev_attr_gpio.attr,
	&dev_attr_period_us.attr,
	&dev_attr_pulse_width_us.attr,
	&dev_attr_enabled.attr,
	NULL,
};

ATTRIBUTE_GROUPS(twinr_servo);

static int __init twinr_servo_init(void)
{
	int ret;

	mutex_init(&twinr_servo.config_lock);
	spin_lock_init(&twinr_servo.state_lock);
	hrtimer_init(&twinr_servo.timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL_PINNED);
	twinr_servo.timer.function = twinr_servo_timer_callback;

	twinr_servo.class = class_create("twinr_servo");
	if (IS_ERR(twinr_servo.class))
		return PTR_ERR(twinr_servo.class);

	twinr_servo.dev = device_create_with_groups(
		twinr_servo.class,
		NULL,
		MKDEV(0, 0),
		&twinr_servo,
		twinr_servo_groups,
		TWINR_SERVO_DEVICE_NAME);
	if (IS_ERR(twinr_servo.dev)) {
		ret = PTR_ERR(twinr_servo.dev);
		class_destroy(twinr_servo.class);
		return ret;
	}

	pr_info("twinr_servo: loaded\n");
	return 0;
}

static void __exit twinr_servo_exit(void)
{
	mutex_lock(&twinr_servo.config_lock);
	twinr_servo_stop_locked(&twinr_servo);
	twinr_servo_request_gpio_locked(&twinr_servo, -1);
	mutex_unlock(&twinr_servo.config_lock);

	device_unregister(twinr_servo.dev);
	class_destroy(twinr_servo.class);
	pr_info("twinr_servo: unloaded\n");
}

module_init(twinr_servo_init);
module_exit(twinr_servo_exit);

MODULE_AUTHOR("OpenAI Codex for Twinr");
MODULE_DESCRIPTION("HRTimer-driven single-servo kernel driver for Twinr");
MODULE_LICENSE("GPL");
