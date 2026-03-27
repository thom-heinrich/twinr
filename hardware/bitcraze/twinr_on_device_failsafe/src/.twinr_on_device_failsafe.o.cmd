cmd_/home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src/twinr_on_device_failsafe.o := arm-none-eabi-gcc -Wp,-MD,/home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src/.twinr_on_device_failsafe.o.d     -I/home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src -D__firmware__ -fno-exceptions -Wall -Wmissing-braces -fno-strict-aliasing -ffunction-sections -fdata-sections -Wdouble-promotion -std=gnu11 -DCRAZYFLIE_FW   -I/tmp/crazyflie-firmware/vendor/CMSIS/CMSIS/Core/Include   -I/tmp/crazyflie-firmware/vendor/CMSIS/CMSIS/DSP/Include   -I/tmp/crazyflie-firmware/vendor/libdw1000/inc   -I/tmp/crazyflie-firmware/vendor/FreeRTOS/include   -I/tmp/crazyflie-firmware/vendor/FreeRTOS/portable/GCC/ARM_CM4F   -I/tmp/crazyflie-firmware/src/config   -I/tmp/crazyflie-firmware/src/platform/interface   -I/tmp/crazyflie-firmware/src/deck/interface   -I/tmp/crazyflie-firmware/src/deck/drivers/interface   -I/tmp/crazyflie-firmware/src/drivers/interface   -I/tmp/crazyflie-firmware/src/drivers/bosch/interface   -I/tmp/crazyflie-firmware/src/drivers/esp32/interface   -I/tmp/crazyflie-firmware/src/hal/interface   -I/tmp/crazyflie-firmware/src/modules/interface   -I/tmp/crazyflie-firmware/src/modules/interface/kalman_core   -I/tmp/crazyflie-firmware/src/modules/interface/lighthouse   -I/tmp/crazyflie-firmware/src/modules/interface/outlierfilter   -I/tmp/crazyflie-firmware/src/modules/interface/cpx   -I/tmp/crazyflie-firmware/src/modules/interface/p2pDTR   -I/tmp/crazyflie-firmware/src/modules/interface/controller   -I/tmp/crazyflie-firmware/src/modules/interface/estimator   -I/tmp/crazyflie-firmware/src/utils/interface   -I/tmp/crazyflie-firmware/src/utils/interface/kve   -I/tmp/crazyflie-firmware/src/utils/interface/lighthouse   -I/tmp/crazyflie-firmware/src/utils/interface/tdoa   -I/tmp/crazyflie-firmware/src/lib/FatFS   -I/tmp/crazyflie-firmware/src/lib/CMSIS/STM32F4xx/Include   -I/tmp/crazyflie-firmware/src/lib/STM32_USB_Device_Library/Core/inc   -I/tmp/crazyflie-firmware/src/lib/STM32_USB_OTG_Driver/inc   -I/tmp/crazyflie-firmware/src/lib/STM32F4xx_StdPeriph_Driver/inc   -I/tmp/crazyflie-firmware/src/lib/vl53l1   -I/tmp/crazyflie-firmware/src/lib/vl53l1/core/inc   -I/home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/build/include/generated -fno-delete-null-pointer-checks -Wno-unused-but-set-variable -Wno-unused-const-variable -fomit-frame-pointer -fno-var-tracking-assignments -Wno-pointer-sign -fno-strict-overflow -fconserve-stack -Werror=implicit-int -Werror=date-time -DCC_HAVE_ASM_GOTO -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -g3 -fno-math-errno -DARM_MATH_CM4 -D__FPU_PRESENT=1 -mfp16-format=ieee -Wno-array-bounds -Wno-stringop-overread -Wno-stringop-overflow -DSTM32F4XX -DSTM32F40_41xxx -DHSE_VALUE=8000000 -DUSE_STDPERIPH_DRIVER -Os -Werror   -c -o /home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src/twinr_on_device_failsafe.o /home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src/twinr_on_device_failsafe.c

source_/home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src/twinr_on_device_failsafe.o := /home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src/twinr_on_device_failsafe.c

deps_/home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src/twinr_on_device_failsafe.o := \
  /usr/include/newlib/math.h \
  /usr/include/newlib/sys/reent.h \
  /usr/include/newlib/_ansi.h \
  /usr/include/newlib/newlib.h \
  /usr/include/newlib/_newlib_version.h \
  /usr/include/newlib/sys/config.h \
    $(wildcard include/config/h//.h) \
  /usr/include/newlib/machine/ieeefp.h \
  /usr/include/newlib/sys/features.h \
  /usr/lib/gcc/arm-none-eabi/10.3.1/include/stddef.h \
  /usr/include/newlib/sys/_types.h \
  /usr/include/newlib/machine/_types.h \
  /usr/include/newlib/machine/_default_types.h \
  /usr/include/newlib/sys/lock.h \
  /usr/include/newlib/sys/cdefs.h \
  /usr/include/newlib/_ansi.h \
  /usr/lib/gcc/arm-none-eabi/10.3.1/include/stdbool.h \
  /usr/lib/gcc/arm-none-eabi/10.3.1/include/stdint.h \
  /usr/include/newlib/string.h \
  /usr/include/newlib/sys/_locale.h \
  /usr/include/newlib/strings.h \
  /usr/include/newlib/sys/string.h \
  /tmp/crazyflie-firmware/src/modules/interface/app.h \
  /tmp/crazyflie-firmware/src/modules/interface/app_channel.h \
  /tmp/crazyflie-firmware/src/modules/interface/crtp.h \
  /tmp/crazyflie-firmware/src/modules/interface/commander.h \
  /tmp/crazyflie-firmware/src/config/config.h \
    $(wildcard include/config/h/.h) \
    $(wildcard include/config/block/address.h) \
  /tmp/crazyflie-firmware/src/drivers/interface/nrf24l01.h \
  /tmp/crazyflie-firmware/src/drivers/interface/nRF24L01reg.h \
  /tmp/crazyflie-firmware/src/config/trace.h \
  /tmp/crazyflie-firmware/src/hal/interface/usec_time.h \
  /tmp/crazyflie-firmware/src/modules/interface/stabilizer_types.h \
  /tmp/crazyflie-firmware/src/hal/interface/imu_types.h \
  /tmp/crazyflie-firmware/src/utils/interface/lighthouse/lighthouse_types.h \
  /tmp/crazyflie-firmware/src/utils/interface/debug.h \
    $(wildcard include/config/debug/print/on/uart1.h) \
  /tmp/crazyflie-firmware/src/modules/interface/console.h \
  /tmp/crazyflie-firmware/src/utils/interface/eprintf.h \
  /usr/lib/gcc/arm-none-eabi/10.3.1/include/stdarg.h \
  /tmp/crazyflie-firmware/vendor/FreeRTOS/include/FreeRTOS.h \
  /tmp/crazyflie-firmware/src/config/FreeRTOSConfig.h \
    $(wildcard include/config/h.h) \
    $(wildcard include/config/debug/queue/monitor.h) \
  /tmp/crazyflie-firmware/src/config/config.h \
  /tmp/crazyflie-firmware/src/utils/interface/cfassert.h \
  /tmp/crazyflie-firmware/vendor/FreeRTOS/include/projdefs.h \
  /tmp/crazyflie-firmware/vendor/FreeRTOS/include/portable.h \
  /tmp/crazyflie-firmware/vendor/FreeRTOS/include/deprecated_definitions.h \
  /tmp/crazyflie-firmware/vendor/FreeRTOS/portable/GCC/ARM_CM4F/portmacro.h \
  /tmp/crazyflie-firmware/vendor/FreeRTOS/include/mpu_wrappers.h \
  /tmp/crazyflie-firmware/src/modules/interface/log.h \
    $(wildcard include/config/debug/log/enable.h) \
  /tmp/crazyflie-firmware/src/modules/interface/param.h \
  /tmp/crazyflie-firmware/src/modules/interface/param_logic.h \
  /tmp/crazyflie-firmware/src/modules/interface/crtp.h \
  /tmp/crazyflie-firmware/src/modules/interface/supervisor.h \
  /tmp/crazyflie-firmware/vendor/FreeRTOS/include/task.h \
  /tmp/crazyflie-firmware/vendor/FreeRTOS/include/list.h \

/home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src/twinr_on_device_failsafe.o: $(deps_/home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src/twinr_on_device_failsafe.o)

$(deps_/home/thh/twinr/hardware/bitcraze/twinr_on_device_failsafe/src/twinr_on_device_failsafe.o):
