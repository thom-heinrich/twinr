obj-m += twinr_servo.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
	$(CC) $(CFLAGS) -o twinr_servo_profile twinr_servo_profile.c

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f twinr_servo_profile
