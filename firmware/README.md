# DWM1001 Multi-Anchor Firmware & UWB ML Toolkit

This repository contains firmware for the Nordic nRF52 + Decawave DW1000 platform that fixes long-standing TDMA and channel impulse response (CIR) issues so a single tag can range sequentially against up to four anchors. Use the initiator build on your tag hardware and the responder build on each anchor to obtain stable, low-latency distance logs.

## Repository layout

| Path | Description |
| --- | --- |
| `main_init.c` | nRF5 entry point for the sequential-polling initiator/tag, including DW1000 bring-up and FreeRTOS tasks. |
| `ss_init_main.c` | Application logic for the initiator: anchor slot scheduler, CIR quality metrics, statistics, and UART logging helpers. |
| `main_resp.c` | nRF5 entry point for the responder/anchor firmware. |
| `ss_resp_main.c` | Responder logic with the slot-based reply calculation and target-ID filtering. |
| `before/` | Reference copies of the unpatched firmware that shipped before the timing, CIR, and filtering fixes. |
| `.vscode/` | Editor settings used for local development (intellisense paths, build tasks, etc.). |

## Firmware

### Features

- Works on DWM1001-DEV (nRF52832 + DW1000) and other boards wired the same way.
- Two binaries: initiator/tag and responder/anchor. The initiator polls anchors sequentially instead of broadcasting; each responder uses a dedicated slot calculated from `ANCHOR_ID`.
- Multi-anchor aware timing: `BASE_REPLY_DELAY_UUS`, `SLOT_SPACING_UUS`, and `GUARD_DELAY_UUS` keep anchors from colliding without enlarging the TDMA frame too much.
- Optional FreeRTOS tasks (define `USE_FREERTOS`) with LED heartbeat for rapid bring-up; bare-metal loop also supported.
- UART prints `DATA,MEAS` records that already include timestamps, SNR-derived RSSI, CIR quality, loss rate, and per-anchor stats for downstream analytics.

### Build prerequisites

1. **Nordic nRF5 SDK 15.x** (the code includes `nrf_drv_clock.h`, `bsp.h`, `boards.h`, etc.).
2. **Decawave DW1000 API** headers and libraries (provides `deca_device_api.h`, `deca_regs.h`, `port_platform.h`).
3. **FreeRTOS** as shipped inside the nRF5 SDK (already wired via `FreeRTOSConfig.h`).
4. SEGGER Embedded Studio, `make`, or any other nRF52-capable toolchain. The sample projects were originally compiled with SES.

### Build and flash

1. Copy `main_init.c`, `ss_init_main.c`, `main_resp.c`, and `ss_resp_main.c` into an nRF5 SDK example folder (for example `examples/dwm1001/uwb_sequential_twr/`).
2. Update the project file or Makefile include paths so both the Nordic SDK (drivers, `components`) and the Decawave library directories are visible.
3. Define the right preprocessor symbols:
   - `USE_FREERTOS` if you want FreeRTOS tasks and scheduler.
   - `DW1000_IRQ`, `TX_ANT_DLY`, and `RX_ANT_DLY` according to your board layout (the defaults match the DWM1001-DEV kit).
4. Build two binaries:
   - **Initiator**: entry point `main_init.c` plus `ss_init_main.c`.
   - **Responder**: entry point `main_resp.c` plus `ss_resp_main.c`. Update `ANCHOR_ID` so each physical anchor owns a unique slot.
5. Flash the images to your tag and anchor boards using `nrfjprog --program <hex> --reset` or the SES "Download & Debug" action.

### Runtime notes

- Connect the tag's UART to a terminal (115200-8-N-1). You should see the DW1000 bring-up logs followed by `DATA,MEAS` lines once polling starts.
- Anchor firmware automatically ignores polls addressed to another anchor; set `target_id` to zero from the initiator to broadcast when debugging.
- `MAX_ANCHORS`, timing slots, and `RNG_DELAY_MS` live in `ss_init_main.c`. Increase `MAX_ANCHORS` if you have more than four devices, but remember to expand the timing window.
- If you run without FreeRTOS, make sure your system clock keeps servicing the DW1000. The bare loop in both `main_init.c` and `main_resp.c` already delays with `nrf_delay_ms`.

## Log format

Initiator firmware lines look like:

```
DATA,MEAS,anchor=2,seq=17,delay_us=1400,timeout_us=3600,dist_m=3.842,rssi_db=-81,loss_pct=8.2,poll_tx=105375912,resp_rx=105380744,resp_tx=105382344,cir=87,method=2,fp_amp2=3477,max_noise=216,std_noise=92,preamble_count=118,snr=5.12,tx_count=142,rx_count=130,status=success
```

CSV parsers can key off `DATA,MEAS` and `DATA,RESULT`. Each line already includes timestamps, slot timing, RF quality indicators, and per-anchor statistics that you can feed into your analytics tool of choice.

## Troubleshooting checklist

- **DW1000 init failures**: verify SPI wires and that `reset_DW1000()` actually pulses the RST pin (some carrier boards tie it to SWD reset).
- **Slot collisions**: confirm every anchor flashed with `ANCHOR_ID` 1..N and that the initiator's `MAX_ANCHORS` matches.
- **Model diverges**: make sure the Excel files were filtered (`filter_type='combo'`) and that the `uwb_data` folder contains only compatible schemas; use `processor.show_dataset_stats()` (add a print) to inspect distributions.
- **Saved models missing**: the script only writes artifacts after a successful evaluation. Check the console for Python exceptions and confirm you have write permissions inside `saved_models/`.

## Credits

- Nordic Semiconductor nRF5 SDK components retain their original license headers inside each C source.
- Decawave DW1000 API headers and helper functions are required to communicate with the UWB radio.
