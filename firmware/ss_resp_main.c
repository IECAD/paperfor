/*! ----------------------------------------------------------------------------
 *  @file    ss_resp_main.c
 *  @brief   Single-sided two-way ranging (SS TWR) responder example code
 *           SIMPLE FIX: Only changed timing values
 */

#include "sdk_config.h"
#include <stdio.h>
#include <string.h>
#include "FreeRTOS.h"
#include "task.h"
#include "deca_device_api.h"
#include "deca_regs.h"
#include "port_platform.h"

/* Unique ID setting for each anchor */
#define ANCHOR_ID 1   // Change to different ID for each anchor (1, 2, 3, 4, etc.)

/* FIXED time slot related parameters */
#define BASE_REPLY_DELAY_UUS    800     // INCREASED: 500 ¡æ 800
#define SLOT_SPACING_UUS        500     // CHANGED: 250 ¡æ 500  
#define GUARD_DELAY_UUS         100     // Same as Initiator

/* Inter-ranging delay period, in milliseconds. */
#define RNG_DELAY_MS 40

/* UWB microsecond (uus) to device time unit (dtu, around 15.65 ps) conversion factor. */
#define UUS_TO_DWT_TIME 65536

/* Speed of light in air, in metres per second. */
#define SPEED_OF_LIGHT 299702547

/* Frames used in the ranging process. */
static uint8 rx_poll_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'W', 'A', 'V', 'E', 0xE0, 0, 0, 0, 0, 0, 0};
static uint8 tx_resp_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'V', 'E', 'W', 'A', 0xE1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* Length of the common part of the message */
#define ALL_MSG_COMMON_LEN 10

/* Index to access some of the fields in the frames involved in the process. */
#define ALL_MSG_SN_IDX 2
#define RESP_MSG_POLL_RX_TS_IDX 10
#define RESP_MSG_RESP_TX_TS_IDX 14
#define RESP_MSG_TS_LEN 4
#define RESP_MSG_ANCHOR_IDX 18  // Anchor ID field index (separate from timestamps)
#define POLL_MSG_TARGET_IDX 10  // Target anchor ID field index

/* Frame sequence number, incremented after each transmission. */
static uint8 frame_seq_nb = 0;

/* Buffer to store received response message. */
#define RX_BUF_LEN 24
static uint8 rx_buffer[RX_BUF_LEN];

/* Hold copy of status register state here for reference so that it can be examined at a debug breakpoint. */
static uint32 status_reg = 0;

/* Timestamps of frames transmission/reception. */
typedef unsigned long long uint64;
static uint64 poll_rx_ts;
static uint64 resp_tx_ts;

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn get_rx_timestamp_u64()
 *
 * @brief Get the RX time-stamp in a 64-bit variable.
 */
static uint64 get_rx_timestamp_u64(void)
{
  uint8 ts_tab[5];
  uint64 ts = 0;
  int i;
  dwt_readrxtimestamp(ts_tab);
  for (i = 4; i >= 0; i--)
  {
    ts <<= 8;
    ts |= ts_tab[i];
  }
  return ts;
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn resp_msg_set_ts()
 *
 * @brief Fill a given timestamp field in the response message with the given value.
 */
static void resp_msg_set_ts(uint8 *ts_field, const uint64 ts)
{
  int i;
  for (i = 0; i < RESP_MSG_TS_LEN; i++)
  {
    ts_field[i] = (ts >> (i * 8)) & 0xFF;
  }
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn calculate_reply_delay()
 *
 * @brief Calculate the expected delay for a specific anchor based on its ID
 *        FIXED VERSION - All anchors use consistent sequential timing
 */
static uint32 calculate_reply_delay(uint8 anchor_id)
{
  // SIMPLE FIX: All anchors follow the same sequential timing rule
  return BASE_REPLY_DELAY_UUS + ((anchor_id - 1) * SLOT_SPACING_UUS) + GUARD_DELAY_UUS;

  /* Fixed calculation results:
   * Anchor 1: 800 + (0*500) + 100 = 900 ¥ìs
   * Anchor 2: 800 + (1*500) + 100 = 1400 ¥ìs  
   * Anchor 3: 800 + (2*500) + 100 = 1900 ¥ìs
   * Anchor 4: 800 + (3*500) + 100 = 2400 ¥ìs
   */
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn main()
 *
 * @brief Application entry point.
 */
int ss_resp_run(void)
{
  /* Activate reception immediately. */
  dwt_rxenable(DWT_START_RX_IMMEDIATE);

  /* Poll for reception of a frame or error/timeout. */
  while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR)))
  {};

  if (status_reg & SYS_STATUS_RXFCG)
  {
    uint32 frame_len;

    /* Clear good RX frame event in the DW1000 status register. */
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG);

    /* A frame has been received, read it into the local buffer. */
    frame_len = dwt_read32bitreg(RX_FINFO_ID) & RX_FINFO_RXFL_MASK_1023;
    if (frame_len <= RX_BUF_LEN)
    {
      dwt_readrxdata(rx_buffer, frame_len, 0);
    }

    /* Check that the frame is a poll sent by "SS TWR initiator" example. */
    rx_buffer[ALL_MSG_SN_IDX] = 0;
    if (memcmp(rx_buffer, rx_poll_msg, ALL_MSG_COMMON_LEN) == 0)
    {
      /* Extract target anchor ID */
      uint8 target_id = rx_buffer[POLL_MSG_TARGET_IDX];

      /* Respond when broadcast (target_id == 0) or when this anchor is target (target_id == ANCHOR_ID) */
      if (target_id == 0 || target_id == ANCHOR_ID)
      {
        uint32 resp_tx_time;
        int ret;

        /* Retrieve poll reception timestamp. */
        poll_rx_ts = get_rx_timestamp_u64();

        /* Calculate response delay time based on anchor ID - Apply FIXED TDMA method */
        uint32 delay_time = calculate_reply_delay(ANCHOR_ID);

        /* Compute final message transmission time. */
        resp_tx_time = (poll_rx_ts + (delay_time * UUS_TO_DWT_TIME)) >> 8;
        dwt_setdelayedtrxtime(resp_tx_time);

        /* Response TX timestamp is the transmission time we programmed plus the antenna delay. */
        resp_tx_ts = (((uint64)(resp_tx_time & 0xFFFFFFFEUL)) << 8) + TX_ANT_DLY;

        /* Write all timestamps in the final message. */
        resp_msg_set_ts(&tx_resp_msg[RESP_MSG_POLL_RX_TS_IDX], poll_rx_ts);
        resp_msg_set_ts(&tx_resp_msg[RESP_MSG_RESP_TX_TS_IDX], resp_tx_ts);

        /* Add anchor ID - keep outside the timestamp fields */
        tx_resp_msg[RESP_MSG_ANCHOR_IDX] = ANCHOR_ID;

        /* Write and send the response message. */
        tx_resp_msg[ALL_MSG_SN_IDX] = frame_seq_nb;
        dwt_writetxdata(sizeof(tx_resp_msg), tx_resp_msg, 0); /* Zero offset in TX buffer. */
        dwt_writetxfctrl(sizeof(tx_resp_msg), 0, 1); /* Zero offset in TX buffer, ranging. */
        ret = dwt_starttx(DWT_START_TX_DELAYED);

        /* If dwt_starttx() returns an error, abandon this ranging exchange and proceed to the next one. */
        if (ret == DWT_SUCCESS)
        {
          /* Poll DW1000 until TX frame sent event set. */
          while (!(dwt_read32bitreg(SYS_STATUS_ID) & SYS_STATUS_TXFRS))
          {};

          /* Clear TXFRS event. */
          dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS);

          /* Increment frame sequence number after transmission of the poll message (modulo 256). */
          frame_seq_nb++;
        }
        else
        {
          /* Reset RX to properly reinitialise LDE operation. */
          dwt_rxreset();
        }
      }
    }
  }
  else
  {
    /* Clear RX error events in the DW1000 status register. */
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_ERR);

    /* Reset RX to properly reinitialise LDE operation. */
    dwt_rxreset();
  }

  return(1);
}

/**@brief SS TWR Responder task entry function.
 *
 * @param[in] pvParameter   Pointer that will be used as the parameter for the task.
 */
void ss_responder_task_function (void * pvParameter)
{
  UNUSED_PARAMETER(pvParameter);

  dwt_setleds(DWT_LEDS_ENABLE);

  while (true)
  {
    ss_resp_run();

    /* Delay a task for a given number of ticks */
    vTaskDelay(RNG_DELAY_MS);

    /* Tasks must be implemented to never return... */
  }
}