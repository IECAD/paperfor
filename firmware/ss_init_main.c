/*! ---------------------------------------------------------------------------- 
 *  @file    ss_init_main.c
 *  @brief   Sequential polling fix - NO BROADCAST, only individual targeting
 *           CIR QUALITY FIXED VERSION - Now provides meaningful CIR values
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "FreeRTOS.h"
#include "task.h"
#include "deca_device_api.h"
#include "deca_regs.h"
#include "port_platform.h"

#define APP_NAME "SS TWR SEQUENTIAL POLLING - CIR FIXED"

/* Inter-ranging delay period, in milliseconds. */
#define RNG_DELAY_MS 40

/* Maximum number of anchors */
#define MAX_ANCHORS 4

/* UWB microsecond (uus) to device time unit (dtu, around 15.65 ps) conversion factor. */
#define UUS_TO_DWT_TIME 65536

/* Speed of light in air, in metres per second. */
#define SPEED_OF_LIGHT 299702547

/* Simple timing parameters */
#define BASE_REPLY_DELAY_UUS    800     
#define SLOT_SPACING_UUS        500     
#define GUARD_DELAY_UUS         100     

/* Single anchor timeout */
#define SINGLE_ANCHOR_TIMEOUT_UUS   2000    // 2ms timeout for single anchor

/* Frames used in the ranging process. */
static uint8 tx_poll_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'W', 'A', 'V', 'E', 0xE0, 0, 0, 0x01, 0x02, 0x03, 0x04};
static uint8 rx_resp_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'V', 'E', 'W', 'A', 0xE1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#define ALL_MSG_COMMON_LEN 10
#define ALL_MSG_SN_IDX 2
#define RESP_MSG_POLL_RX_TS_IDX 10
#define RESP_MSG_RESP_TX_TS_IDX 14
#define RESP_MSG_TS_LEN 4
#define RESP_MSG_ANCHOR_IDX 18  // Anchor ID field index (separate from timestamps)
#define POLL_MSG_TARGET_IDX 10

/* Frame sequence number */
static uint8 frame_seq_nb = 0;

/* Current anchor being polled (1, 2, 3, 4 in sequence) */
static uint8 current_anchor = 1;

/* Buffer to store received response message. */
#define RX_BUF_LEN 24
static uint8 rx_buffer[RX_BUF_LEN];

static uint32 status_reg = 0;
static double tof;
static double distance;

/* Distance storage array by anchor */
static double anchor_distances[MAX_ANCHORS+1];
static int8 anchor_rssi[MAX_ANCHORS+1];

/* Structure for statistics tracking */
typedef struct {
  uint32_t tx_count;
  uint32_t rx_count;
  float success_rate;
} anchor_stats_t;

static anchor_stats_t anchor_stats[MAX_ANCHORS+1];

/* Transaction Counters */
static volatile int tx_count = 0;
static volatile int rx_count = 0;

/*! ------------------------------------------------------------------------------------------------------------------ 
 * @fn calculate_reply_delay() 
 */
static uint32 calculate_reply_delay(uint8 anchor_id){
  return BASE_REPLY_DELAY_UUS + ((anchor_id - 1) * SLOT_SPACING_UUS) + GUARD_DELAY_UUS;
}

/*! ------------------------------------------------------------------------------------------------------------------ 
 * @fn resp_msg_get_ts() 
 */
static void resp_msg_get_ts(uint8 *ts_field, uint32 *ts){
  int i;
  *ts = 0;
  for (i = 0; i < RESP_MSG_TS_LEN; i++)
  {
    *ts += ts_field[i] << (i * 8);
  }
}

/*! ------------------------------------------------------------------------------------------------------------------ 
 * @fn get_rx_signal_level() 
 */
static int8 get_rx_signal_level(uint16 fp_ampl2){
  int8 rxLevel;
  if (fp_ampl2 < 0x4000) {
    rxLevel = (int8)(-100 + 10 * log10f((float)fp_ampl2) + 6);
  } else {
    rxLevel = (int8)(-100 + 6 + 10 * log10f(0x4000) + 10 * log10f((float)fp_ampl2/0x4000));
  }
  return rxLevel;
}

/*! ------------------------------------------------------------------------------------------------------------------ 
 * @fn calculate_loss_rate() 
 */
static float calculate_loss_rate(uint32_t tx_count, uint32_t rx_count){
  if (tx_count == 0) return 0.0f;
  return 100.0f * (1.0f - ((float)rx_count / (float)tx_count));
}

/*! ------------------------------------------------------------------------------------------------------------------ 
 * @fn get_measurement_timestamp_ms()
 * @brief Return milliseconds since boot (approximate when RTOS not used)
 */
static uint32 get_measurement_timestamp_ms(void){
#ifdef USE_FREERTOS
  return (uint32)(xTaskGetTickCount() * portTICK_PERIOD_MS);
#else
  static uint32 timestamp_ms = 0;
  timestamp_ms += RNG_DELAY_MS;
  return timestamp_ms;
#endif
}

typedef struct {
  int year;
  int month;
  int day;
  int hour;
  int minute;
  int second;
} datetime_t;

static datetime_t g_base_datetime;
static bool g_datetime_initialized = false;

static int month_from_str(const char *str){
  static const char *months[] = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};
  for (int i = 0; i < 12; i++){
    if (strncmp(str, months[i], 3) == 0){
      return i + 1;
    }
  }
  return 1;
}

static bool is_leap_year(int year){
  return ((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0);
}

static int days_in_month(int year, int month){
  static const int days_per_month[] = {31,28,31,30,31,30,31,31,30,31,30,31};
  int days = days_per_month[month - 1];
  if (month == 2 && is_leap_year(year)){
    days += 1;
  }
  return days;
}

static void add_seconds(datetime_t *dt, uint32 seconds){
  if (seconds == 0) return;

  dt->second += (int)(seconds % 60U);
  seconds /= 60U;
  if (dt->second >= 60){
    dt->second -= 60;
    seconds += 1;
  }

  dt->minute += (int)(seconds % 60U);
  seconds /= 60U;
  if (dt->minute >= 60){
    dt->minute -= 60;
    seconds += 1;
  }

  dt->hour += (int)(seconds % 24U);
  seconds /= 24U;
  if (dt->hour >= 24){
    dt->hour -= 24;
    seconds += 1;
  }

  while (seconds > 0){
    int dim = days_in_month(dt->year, dt->month);
    dt->day += 1;
    if (dt->day > dim){
      dt->day = 1;
      dt->month += 1;
      if (dt->month > 12){
        dt->month = 1;
        dt->year += 1;
      }
    }
    seconds -= 1;
  }
}

static void init_base_datetime(void){
  if (g_datetime_initialized){
    return;
  }
  const char *date_str = __DATE__;
  const char *time_str = __TIME__;

  char month_str[4];
  int day, year;
  sscanf(date_str, "%3s %d %d", month_str, &day, &year);

  int hour, minute, second;
  sscanf(time_str, "%d:%d:%d", &hour, &minute, &second);

  g_base_datetime.year = year;
  g_base_datetime.month = month_from_str(month_str);
  g_base_datetime.day = day;
  g_base_datetime.hour = hour;
  g_base_datetime.minute = minute;
  g_base_datetime.second = second;
  g_datetime_initialized = true;
}

static void format_timestamp(uint32 timestamp_ms, char *buffer, size_t buffer_len){
  init_base_datetime();

  datetime_t current = g_base_datetime;
  uint32 elapsed_seconds = timestamp_ms / 1000U;
  uint32 elapsed_ms = timestamp_ms % 1000U;
  add_seconds(&current, elapsed_seconds);

  snprintf(buffer,
           buffer_len,
           "%04d-%02d-%02d %02d-%02d-%02d.%03lu",
           current.year,
           current.month,
           current.day,
           current.hour,
           current.minute,
           current.second,
           (unsigned long)elapsed_ms);
}

static uint8 calculate_cir_quality(uint8 anchor_id, dwt_rxdiag_t *rx_diag, uint8 *method_used, float *snr_out){
  uint8 cir_quality = 0;
  uint16 fp_ampl2 = rx_diag->firstPathAmp2;
  uint16 max_noise = rx_diag->maxNoise;

  if (method_used != NULL) {
    *method_used = 0;
  }
  if (snr_out != NULL) {
    *snr_out = 0.0f;
  }

  if (max_noise > 0 && fp_ampl2 > 0) {
    float signal_power = (float)fp_ampl2;
    float noise_power = (float)max_noise;
    float snr_ratio = signal_power / noise_power;

    if (snr_ratio > 16.0f) snr_ratio = 16.0f;
    if (snr_ratio < 1.0f) snr_ratio = 1.0f;

    float normalized = (snr_ratio - 1.0f) / 15.0f;
    cir_quality = (uint8)(normalized * 255.0f);

    if (method_used != NULL) {
      *method_used = 1;
    }
    if (snr_out != NULL) {
      *snr_out = snr_ratio;
    }
    return cir_quality;
  }

  uint16 std_noise = rx_diag->stdNoise;
  if (std_noise > 0 && fp_ampl2 > 0) {
    float signal_power = (float)fp_ampl2;
    float noise_power = (float)std_noise;
    float snr_ratio = signal_power / noise_power;

    if (snr_ratio > 32.0f) snr_ratio = 32.0f;
    if (snr_ratio < 2.0f) snr_ratio = 2.0f;

    float normalized = (snr_ratio - 2.0f) / 30.0f;
    cir_quality = (uint8)(normalized * 255.0f);

    if (method_used != NULL) {
      *method_used = 2;
    }
    if (snr_out != NULL) {
      *snr_out = snr_ratio;
    }
    return cir_quality;
  }

  uint16 pream_count = rx_diag->rxPreamCount;
  if (pream_count > 0) {
    if (pream_count > 1000) {
      cir_quality = 255;
    } else if (pream_count > 500) {
      cir_quality = 200;
    } else if (pream_count > 100) {
      cir_quality = 150;
    } else if (pream_count > 50) {
      cir_quality = 100;
    } else {
      cir_quality = 50;
    }

    if (method_used != NULL) {
      *method_used = 3;
    }
    return cir_quality;
  }

  if (method_used != NULL) {
    *method_used = 0;
  }
  return 0;
}

/*! ------------------------------------------------------------------------------------------------------------------ 
 * @fn poll_single_anchor() 
 * @brief Poll a single specific anchor and wait for response 
 */
static bool poll_single_anchor(uint8 anchor_id){
  // Set target anchor ID (NO BROADCAST - only specific target)
  tx_poll_msg[POLL_MSG_TARGET_IDX] = anchor_id;
    
  // Set timeout for single anchor response
  uint32 expected_delay = calculate_reply_delay(anchor_id);
  uint32 timeout = expected_delay + SINGLE_ANCHOR_TIMEOUT_UUS;
  dwt_setrxtimeout(timeout);
  
  tx_poll_msg[ALL_MSG_SN_IDX] = frame_seq_nb;
  dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS);
  dwt_writetxdata(sizeof(tx_poll_msg), tx_poll_msg, 0);
  dwt_writetxfctrl(sizeof(tx_poll_msg), 0, 1);
  dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);
  
  tx_count++;
  anchor_stats[anchor_id].tx_count++;

  get_measurement_timestamp_ms();

  // Wait for response or timeout
  while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR)))
  {};
  
  frame_seq_nb++;
  
  if (status_reg & SYS_STATUS_RXFCG)
  {
    uint32 frame_len;
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG);
    frame_len = dwt_read32bitreg(RX_FINFO_ID) & RX_FINFO_RXFLEN_MASK;
    if (frame_len <= RX_BUF_LEN)
    {
      dwt_readrxdata(rx_buffer, frame_len, 0);
    }
    
    rx_buffer[ALL_MSG_SN_IDX] = 0;
    if (memcmp(rx_buffer, rx_resp_msg, ALL_MSG_COMMON_LEN) == 0)
    {
      uint8 responding_anchor = rx_buffer[RESP_MSG_ANCHOR_IDX];
      if (responding_anchor == anchor_id)
      {
        rx_count++;
        anchor_stats[anchor_id].rx_count++;
        
        uint32 poll_tx_ts, resp_rx_ts, poll_rx_ts, resp_tx_ts;
        int32 rtd_init, rtd_resp;
        
        poll_tx_ts = dwt_readtxtimestamplo32();
        resp_rx_ts = dwt_readrxtimestamplo32();
        resp_msg_get_ts(&rx_buffer[RESP_MSG_POLL_RX_TS_IDX], &poll_rx_ts);
        resp_msg_get_ts(&rx_buffer[RESP_MSG_RESP_TX_TS_IDX], &resp_tx_ts);
        
        rtd_init = resp_rx_ts - poll_tx_ts;
        rtd_resp = resp_tx_ts - poll_rx_ts;
        tof = ((rtd_init - rtd_resp) / 2.0f) * DWT_TIME_UNITS;
        
        if (tof < 0) {
          tof = fabs(tof);
        }
        
        distance = tof * SPEED_OF_LIGHT;
        anchor_distances[anchor_id] = distance;
        
        dwt_rxdiag_t rx_diag;
        dwt_readdiagnostics(&rx_diag);
        int8 rssi = get_rx_signal_level(rx_diag.firstPathAmp2);
        anchor_rssi[anchor_id] = rssi;

        float loss_rate = calculate_loss_rate(anchor_stats[anchor_id].tx_count, anchor_stats[anchor_id].rx_count);

        uint8 method_used = 0;
        float cir_snr = 0.0f;
        uint8 cir_quality = calculate_cir_quality(anchor_id, &rx_diag, &method_used, &cir_snr);
        uint8 seq_id = (uint8)(frame_seq_nb - 1);

        printf("DATA,MEAS,anchor=%u,seq=%u,delay_us=%u,timeout_us=%u,dist_m=%.3f,rssi_db=%d,loss_pct=%.1f",
               anchor_id,
               (unsigned int)seq_id,
               expected_delay,
               timeout,
               distance,
               rssi,
               loss_rate);
        printf(",poll_tx=%u,resp_rx=%u,resp_tx=%u,cir=%u,method=%u,fp_amp2=%u,max_noise=%u",
               poll_tx_ts,
               resp_rx_ts,
               resp_tx_ts,
               cir_quality,
               method_used,
               rx_diag.firstPathAmp2,
               rx_diag.maxNoise);
        printf(",std_noise=%u,preamble_count=%u,snr=%.2f,tx_count=%d,rx_count=%d,status=success\r\n",
               rx_diag.stdNoise,
               rx_diag.rxPreamCount,
               cir_snr,
               anchor_stats[anchor_id].tx_count,
               anchor_stats[anchor_id].rx_count);

        return true;  // Success
      }
      else
      {
        printf("DATA,RESULT,anchor=%u,status=unexpected_anchor,received=%u\r\n",
               anchor_id,
               responding_anchor);
      }
    }
  }
  else
  {
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
    dwt_rxreset();
        
    if (status_reg & SYS_STATUS_ALL_RX_TO) {
      printf("DATA,RESULT,anchor=%u,status=timeout\r\n", anchor_id);
    } else {
      printf("DATA,RESULT,anchor=%u,status=rx_error\r\n", anchor_id);
    }
  }

  return false;  // Failed
}

/*! ------------------------------------------------------------------------------------------------------------------ 
 * @fn main() 
 * @brief Application entry point - Sequential polling main loop 
 */
int ss_init_run(void){
  // Poll current anchor
  poll_single_anchor(current_anchor);
    
  // Move to next anchor in sequence
  current_anchor++;
  if (current_anchor > MAX_ANCHORS) {
    current_anchor = 1;  // Loop back to anchor 1
    printf("\r\n\r\n");
  }
  
  return(1);
}

/**@brief SS TWR Initiator task entry function. */
void ss_initiator_task_function (void * pvParameter){
  UNUSED_PARAMETER(pvParameter);
  dwt_setleds(DWT_LEDS_ENABLE);
  
  // Initialize statistics
  for (int i = 0; i <= MAX_ANCHORS; i++) {
    anchor_distances[i] = 0;
    anchor_rssi[i] = 0;
    anchor_stats[i].tx_count = 0;
    anchor_stats[i].rx_count = 0;
    anchor_stats[i].success_rate = 0;
  }
  
  printf("Starting %s\r\n", APP_NAME);
  printf("Sequential polling mode - NO BROADCAST\r\n");
  printf("CIR Quality calculation - FIXED VERSION\r\n");
  printf("Anchor delays: ");
  for (int i = 1; i <= MAX_ANCHORS; i++) {
    printf("A%d:%dus ", i, calculate_reply_delay(i));
  }
  printf("\r\n\r\n");
  
  vTaskDelay(10);
  
  while (true)
  {
    ss_init_run();
    vTaskDelay(RNG_DELAY_MS);
  }
}
