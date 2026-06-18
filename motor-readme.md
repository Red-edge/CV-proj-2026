# Motor README

This project currently controls the yaw motor through Linux SocketCAN on `can0`
at `1 Mbps`. The configured yaw motor CAN id is `0x8001`, sent as an extended
CAN frame.

## What Is Clear

- The project motor model is documented internally as `HTDW-5047-36-NE`.
- The official SDK source checked for this note is
  `HighTorque-Robotics/livelybot_hardware_sdk`, local commit `fd09ab0`
  (`2025-06-19`, "update 4.3.3 electronic-control SDK").
- Existing Python and C++ code agree on the main speed-control frame:

```text
id:   0x8001 extended
data: 07 35 <speed_i16_le> <torque_i16_le> <pos_u16_le>
```

- Current code uses:

```text
speed_raw = rpm / 0.015
torque_raw = 2000
pos_raw = 0x8000
```

- Stop frame used by existing code and tests:

```text
id:   0x8001 extended
data: 01 00 00
```

- Brake frame present in old Python tools:

```text
id:   0x8001 extended
data: 01 00 0F
```

- Status query frame from the HighTorque/HTDW control guide:

```text
id:   0x8001 extended
data: 17 01
```

- Expected status response, observed on this setup with CAN id `0x100`:

```text
data: 27 01 <pos_i16_le> <vel_i16_le> <torque_i16_le>
pos_turns = pos_raw * 0.0001
vel_rpm   = vel_raw * 0.00025 * 60
```

The C++ yaw controller now sends `17 01` after speed commands and parses `27 01`
status feedback from either the command id or `0x100`.

## Official HighTorque SDK Notes

Official repository:

```text
https://github.com/HighTorque-Robotics/livelybot_hardware_sdk
```

The SDK is organized around a `livelybot_serial::robot` object rather than raw
SocketCAN frames:

```text
livelybot_serial::robot rb;
rb.Motors[N]->velocity(...);
rb.Motors[N]->pos_vel_MAXtqe(...);
rb.motor_send_2();
```

Important SDK files and their meaning:

```text
readme.md
  Motor control entry points:
  fresh_cmd_int16, position, velocity, torque, pos_vel_MAXtqe,
  pos_vel_tqe_kp_kd, pos_vel_kp_kd, then rb.motor_send_2().

src/livelybot_bringup/src/motor_set_zero.cpp
  Official zero-reset example. Calls rb.set_reset_zero(), then queries state.

src/livelybot_bringup/src/motor_move_zero.cpp
  Official move-to-zero example. Uses m->pos_vel_MAXtqe(0, 0.3, 100).

src/livelybot_bringup/src/motor_feedback.cpp
  Official feedback example. Calls rb.send_get_motor_state_cmd().

src/livelybot_serial/include/serial_struct.h
  Board/serial command modes:
  MODE_RESET_ZERO=0x01, MODE_CONF_WRITE=0x02, MODE_STOP=0x03,
  MODE_MOTOR_STATE=0x06, MODE_CONF_LOAD=0x07, MODE_RUNZERO=0x09,
  MODE_VELOCITY=0x81, MODE_POS_VEL_TQE=0x90.

src/livelybot_serial/src/hardware/robot.cc
  rb.set_reset_zero({motor_index}) restores config, resets zero, then saves.

src/livelybot_serial/src/hardware/canport.cc
  Implements set_conf_load, set_reset_zero, set_conf_write and
  send_get_motor_state_cmd as CANboard/serial wrapper commands.

src/livelybot_serial/src/hardware/motor.cc
  Implements velocity and pos_vel_MAXtqe by filling the SDK command cache.
```

The important distinction is that the SDK commands above are board/serial
commands sent through the SDK transport. They are not the same API layer as the
current `cvproj_capture` yaw controller, which opens Linux `can0` directly and
writes raw CAN frames to the motor.

## Official SDK Zero-Reset Flow

The official whole-robot zero example calls:

```text
rb.set_reset_zero()
```

At the CANport layer this sends `MODE_RESET_ZERO` with data `0x7f`, then waits
until all configured motor ids acknowledge the reset.

For a specified motor, the SDK does a stronger three-step sequence:

```text
set_conf_load(motor_id)
set_reset_zero(motor_id)
set_conf_write(motor_id)
```

That means:

```text
1. Restore settings from flash.
2. Set the current motor position as zero.
3. Save settings back to flash.
```

The current project zero script is a direct-CAN script. It sends the
HighTorque/HTDW motor-level `set_zero` and `save` frames directly over
SocketCAN, but it does not use the SDK `livelybot_serial::robot` transport and
does not currently perform the SDK's explicit `conf_load` step. This is a
known difference, not a hidden assumption.

## Zero Reset

Recommended current workflow: use a software zero stored in the runtime config,
not the motor flash zero. The direct-CAN `conf_write` path currently returns
`30 B3 02 01` on this setup, so the motor internal zero is not reliable enough
for the full aiming program.

Record the current yaw pose as the software center:

```bash
cd ~/workspace/CV-proj-2026
# manually place yaw at optical/mechanical center first
./scripts/motor_reset_zero.sh --record-software-zero configs/t265_yaw_real.conf
```

This command only sends the read-status frame `17 01`. It does not send
`set_zero`, `save`, torque, or speed commands. It writes:

```text
yaw-feedback-zero-deg=<current motor feedback angle>
```

to `configs/t265_yaw_real.conf`. The C++ yaw controller then uses:

```text
relative_yaw_deg = raw_motor_feedback_deg - yaw-feedback-zero-deg
```

All `+/-30 deg` safety checks and target tracking are applied to
`relative_yaw_deg`. This means the physical pose recorded by the script becomes
runtime yaw `0 deg`, even if the motor internal zero remains offset.

Latest software-zero recording on this setup:

```text
raw feedback: pos=-0.15000 turns (-54.00 deg)
config: yaw-feedback-zero-deg=-54.000000
```

The dedicated zero-reset script is:

```bash
cd ~/workspace/CV-proj-2026
./scripts/motor_reset_zero.sh
```

Safe status-only check:

```bash
./scripts/motor_reset_zero.sh --status-only
```

Before running it, manually place the yaw axis at the mechanical/optical center
you want to treat as zero. The script sends:

```text
stop:      01 00 00
set_zero:  40 01 04 64 20 63 0A
save:      05 B3 02 00 00
status:    17 01
stop:      01 00 00
```

`set_zero` and `save` come from the HighTorque/HTDW CAN control guide for
setting the current motor position as zero and saving settings. If the script
receives a `27 01` status frame after reset, it prints parsed position and
velocity.

The script now follows the official direct-CAN timing more closely:

```text
rezero_pos: 40 01 04 64 20 63 0A
wait:       about 1000 ms
conf_write: 05 B3 02 00 00
verify:     17 01, position must be close to 0 deg
```

The official guide also recommends power-cycling the motor after `conf_write`.
If `./scripts/motor_reset_zero.sh` reports that feedback was received but the
position is still far from `0 deg`, power-cycle the motor and run
`./scripts/motor_reset_zero.sh --status-only` again before starting the full
program.

## Motor Diagnostics And Protection Check

Use this before zero reset or full automatic aiming:

```bash
cd ~/workspace/CV-proj-2026
./scripts/motor_diag.sh
```

The diagnostic script:

- brings up `can0` at `1 Mbps`;
- sends `stop`;
- sends the reply-enabled status query `17 01` to id `0x8001`;
- parses `27 01` feedback from id `0x100`;
- optionally scans a conservative read-only register range with `--scan`;
- optionally tries periodic status return with `--timed-return-ms 100`, then
  disables it before exit.

Latest protection/status check:

```text
can0 state ERROR-ACTIVE
bus-errors=0 error-warn=0 error-pass=0 bus-off=0
TX stop -> RX id=0x100 dlc=0
TX 17 01 -> RX id=0x100 data=27 01 1F FA 00 00 00 00
pos=-0.15050 turns (-54.18 deg), vel=+0.00 rpm, torque_raw=0
```

This means the motor is replying and the bus is not in CAN error protection.
The basic feedback does not show motion or torque load. The current problem is
still the internal zero offset, not a proven bus-off or torque-protection state.

Latest direct-CAN zero-reset result:

```text
TX set_zero 40 01 04 64 20 63 0A
RX 41 01 00
TX save 05 B3 02 00 00
RX 30 B3 02 01
TX status 17 01
RX 27 01 23 FA FF FF 00 00
pos=-0.15010 turns (-54.04 deg)
```

Interpretation:

- `41 01 00` is treated as a `rezero_pos` acknowledgement with code `0`.
- `30 B3 02 01` is treated as a `conf_write` failure/unsupported response.
- The status position did not move close to zero after the reset attempt.

So the current evidence is not "no motor response"; the motor responds and
accepts the zero command, but the direct-CAN save/effective-zero path is not
completing on this setup. Next check after this output is:

```bash
# physically power-cycle the motor, then:
cd ~/workspace/CV-proj-2026
./scripts/motor_reset_zero.sh --status-only
```

If the position is still around `-54 deg` after a power cycle, stop repeating
the direct-CAN zero script and use the official SDK/CANboard
`conf_load -> set_reset_zero -> conf_write` flow, because direct `conf_write`
is returning `30 B3 02 01`.

`--scan` also returned some unsupported-register replies such as
`31 xx 01`; these are treated as protocol-level read failures for unsupported
addresses, not as confirmed motor protection faults. Periodic return command
`05 B4 02 64 00` currently replies `30 B4 02 01` and does not start periodic
`27 01` streaming on this setup, so use single-shot `17 01` status queries for
verification.

On the current hardware, `--status-only` returned:

```text
id=0x100 data=27 01 1F FA 00 00 00 00
pos=-0.15050 turns (-54.18 deg), vel=+0.00 rpm
```

That means the motor can report internal position, and the current internal
zero is not aligned with the current physical yaw pose. Only run the zero reset
when the yaw axis is manually placed at the desired center.

After adding the zero-window protection, a real short run before zero reset
produced:

```text
yaw_current_deg=-51.336000
yaw_rpm=0.000000
yaw_status=outside-zero-window-test-target
```

So the full program now detects a bad zero window and holds instead of moving.

## Running The Full T265 Yaw Program

After recording the software zero:

```bash
cd ~/workspace/CV-proj-2026
./scripts/motor_reset_zero.sh --record-software-zero configs/t265_yaw_real.conf
./scripts/run_t265_yaw_real.sh
```

The default config is:

```text
configs/t265_yaw_real.conf
```

Relevant safety limits:

```text
yaw-enable=true
yaw-can-iface=can0
yaw-id=0x8001
yaw-feedback-zero-deg=-54.000000
yaw-max-angle=30       # absolute software boundary, deg
yaw-limit-margin-deg=2 # safety margin before the boundary, deg
yaw-max-rpm=40   # maximum output speed in rpm
yaw-kp=6
yaw-deadband=0.35
```

The C++ controller treats `yaw-feedback-zero-deg` as the centered pose.
`yaw-max-angle=30` is the absolute software boundary. Any requested target past
that range is clamped to `+/-30 deg`. `yaw-limit-margin-deg=2` does not change
the final boundary; it reduces the allowed rpm as the feedback angle approaches
the boundary, so the controller does not actively push past it. If feedback is
already outside `+/-30 deg`, the controller forces the target to the nearest
boundary and only allows inward recovery motion.

If startup feedback reports the motor is already outside the `+/-30 deg`
software-zero-centered window, the controller refuses to track and sends
`0 rpm` with status:

```text
outside-zero-window
```

This protects the system from using an old or wrong software zero as the center.
Re-record the software zero first, then run the full program.

## Current Driver Evaluation

Current status: usable for the present single-yaw direct-CAN setup after the
motor zero has been correctly reset, but not a full official-SDK-equivalent
driver.

What is already normal enough for this project:

- The program opens `can0` at the Linux SocketCAN layer and sends extended id
  `0x8001` speed frames.
- The speed frame format, stop frame, status query and `27 01` feedback parser
  are internally consistent with the existing project tools and live bus
  behavior.
- `candump` has observed the real program transmitting speed frames and stop
  frames on `can0`.
- The C++ controller reads current position feedback and refuses motion when
  the motor reports a yaw angle outside the configured `+/-30 deg` zero window.
- The configured software hard boundary is now `yaw-max-angle=30`; normal
  commands use `yaw-max-angle - yaw-limit-margin-deg`, and output rpm is also
  clamped before speed is sent.

What is different from the official SDK:

- The project does not link or instantiate `livelybot_serial::robot`.
- The project does not call SDK helpers such as `m->velocity(...)`,
  `m->pos_vel_MAXtqe(...)`, `rb.motor_send_2()`, `rb.set_reset_zero(...)`, or
  `rb.send_get_motor_state_cmd()`.
- The project bypasses the SDK CANboard/serial command modes and sends raw
  motor CAN frames directly.
- The zero script sends direct `set_zero` and `save` frames, while the SDK
  specified-motor flow is `conf_load -> reset_zero -> conf_write`.
- SDK-side config protection (`pos_limit_enable`, `pos_upper`, `pos_lower`) is
  not being used by `cvproj_capture`; the current protection is implemented in
  the C++ yaw controller itself.

Practical conclusion:

```text
Yes, the current code can drive the yaw motor through direct SocketCAN, and it
now has a feedback-based guard that prevents movement if the stored internal
zero is wrong.

No, it should not be described as an official SDK driver yet. It is a direct
CAN compatibility layer for the current single-yaw hardware.
```

Before the next real full-function run, do this once:

```bash
cd ~/workspace/CV-proj-2026
./scripts/motor_reset_zero.sh --status-only
# manually place yaw at optical/mechanical center, then:
./scripts/motor_reset_zero.sh --record-software-zero configs/t265_yaw_real.conf
./scripts/run_t265_yaw_real.sh
```

After recording, the config should contain `yaw-feedback-zero-deg=<current raw
angle>`. The raw `--status-only` value no longer needs to be close to `0 deg`;
the program subtracts the recorded value at runtime.

## Limit Sweep Script

After confirming the yaw axis has enough physical clearance, run:

```bash
cd ~/workspace/CV-proj-2026
./scripts/motor_limit_sweep.sh configs/t265_yaw_real.conf
```

The script reads these config keys:

```text
yaw-feedback-zero-deg
yaw-max-angle
yaw-max-rpm
yaw-can-iface
yaw-id
```

It uses the configured `yaw-max-rpm` as the maximum output speed in rpm, then
executes:

```text
software zero -> left hard limit (+yaw-max-angle) -> right hard limit (-yaw-max-angle) -> software zero
```

If the motor starts outside the hard range, the script first recovers it to the
nearest `+/-30 deg` boundary, then continues the sweep.

Direction convention for the current installation:

```text
left turn == feedback position increases == positive relative yaw
```

For a non-moving check:

```bash
./scripts/motor_limit_sweep.sh configs/t265_yaw_real.conf --dry-run
```

## PID Tuning Script

PID parameters live in the same runtime config and are re-read every time the
main program or PID tuning script starts:

```text
yaw-max-rpm=10
yaw-kp=6
yaw-ki=0
yaw-kd=0
yaw-deadband=0.35
```

Units:

```text
yaw-max-rpm: maximum output speed, rpm
yaw-kp: rpm per degree
yaw-ki: rpm per degree-second
yaw-kd: rpm per degree per second
```

Dedicated tuning command:

```bash
cd ~/workspace/CV-proj-2026
./scripts/motor_pid_tune.sh configs/t265_yaw_real.conf
```

Dry run:

```bash
./scripts/motor_pid_tune.sh configs/t265_yaw_real.conf --dry-run
```

The tuning script is intentionally different from the safe full program. It
uses the software zero and PID parameters from config, then commands timed
target coordinates:

```text
0 deg -> +yaw-max-angle -> -yaw-max-angle -> 0 deg
```

Each target is held for `1.5 s` by default (`--segment-s 1.5`). The script does
not wait for the motor to arrive before moving to the next target. It also
intentionally bypasses the main program's limit recovery and boundary braking
logic, so use it only when the yaw axis has enough physical clearance. It writes
a CSV under `outputs/pid_tune_*.csv` and an SVG plot with the same timestamp.
The CSV contains time, target, relative angle, error, command rpm, feedback rpm,
raw CAN feedback id/payload, `pos_raw`, `vel_raw`, torque and PID values. Use
the SVG feedback curve to find the critical oscillation point while sweeping
`yaw-kp`, `yaw-ki`, `yaw-kd` and `yaw-max-rpm`.

Custom output paths:

```bash
./scripts/motor_pid_tune.sh configs/t265_yaw_real.conf \
  --output outputs/kp6_ki0_kd0.csv \
  --plot outputs/kp6_ki0_kd0.svg
```

## Current Test Evidence

The T265 backend has been verified with:

```text
Intel RealSense T265
Fisheye 1: 848x800 @ 30Hz Y8
fisheye 1 intrinsics: fx=286.064 fy=286.063 cx=432.019 cy=392.678
```

Real CAN speed-frame test used:

```bash
./scripts/run_t265_yaw_real.sh configs/t265_yaw_real.conf \
  --detector none \
  --yaw-test-target-x 460 \
  --yaw-max-rpm 15 \
  --yaw-kp 3 \
  --max-frames 20
```

Observed by `candump`:

```text
can0 00008001#0735E803D0070080
can0 100#
...
can0 00008001#07350000D0070080
can0 00008001#010000
```

The speed frame above commands about `15 rpm`, then the program sends zero speed
and stop on exit.

## What Is Not Fully Clear Yet

- The command, stop, status-query, internal-zero, save, and feedback parsing
  paths are now clear enough to operate and verify.
- The official SDK confirms the intended high-level zero/control/status APIs,
  but those APIs are wrapped by the SDK board/serial transport. We have not yet
  ported that SDK transport into this project.
- The direct-CAN zero script has not been proven to be byte-for-byte equivalent
  to the SDK `conf_load -> reset_zero -> conf_write` path, because the SDK
  commands are not exposed as raw `can0` frames in the same layer.
- We have not yet proven whether `0x100` is a shared response id for both yaw
  and pitch motors. If pitch is added later, feedback association should be
  checked with one motor at a time.
- The full manufacturer direct-CAN manual is still not present in the repo; the
  direct `set_zero`/`save` frames are based on the HighTorque/HTDW guide and
  live bus behavior, while the official livelybot SDK documents the higher
  transport layer.

The safe conclusion is: yaw speed control, software-zero recording, status query,
and +/-30 deg software limiting are clear enough for the current single-yaw
setup after zero has been reset. Multi-axis feedback routing and full
official-SDK parity remain the main unknowns.
