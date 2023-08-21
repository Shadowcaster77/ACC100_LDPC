/**
 * @file test_ldpc_baseband.cc
 * @brief Test LDPC performance after encoding, modulation, demodulation,
 * and decoding when different levels of
 * Gaussian noise is added to CSI
 */
#include "rte_bbdev.h"
#include "rte_bbdev_op.h"
#include "rte_bus_vdev.h"
#include "rte_lcore.h"
#include "rte_eal.h"
#include "rte_mbuf.h"
#include "rte_mempool.h"



#include <gflags/gflags.h>
#include <immintrin.h>

#include <bitset>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include "armadillo"
#include "comms-lib.h"
#include "config.h"
#include "data_generator.h"
#include "datatype_conversion.h"
#include "gettime.h"
#include "memory_manage.h"
#include "modulation.h"
#include "phy_ldpc_decoder_5gnr.h"
#include "utils_ldpc.h"

#include <netinet/ether.h>
#include <rte_byteorder.h>
#include <rte_cycles.h>
#include <rte_debug.h>
#include <rte_distributor.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_flow.h>
#include <rte_ip.h>
#include <rte_malloc.h>
#include <rte_pause.h>
#include <rte_prefetch.h>
#include <rte_udp.h>
#include <unistd.h>


#define TEST_SUCCESS    0
#define TEST_FAILED     -1
#define TEST_SKIPPED    1
#define MAX_QUEUES RTE_MAX_LCORE
#define OPS_CACHE_SIZE 256U
#define MAX_PKT_BURST 32


#define TEST_ASSERT_SUCCESS(val, msg, ...) do { \
		typeof(val) _val = (val); \
		if (!(_val == 0)) { \
			printf("TestCase %s() line %d failed (err %d): " \
				msg "\n", __func__, __LINE__, _val, \
				##__VA_ARGS__); \
			return TEST_FAILED; \
		} \
} while (0)


static struct active_device {
	const char *driver_name;
	uint8_t dev_id;
	uint16_t supported_ops;
	uint16_t queue_ids[MAX_QUEUES];
	uint16_t nb_queues;
	struct rte_mempool *ops_mempool;
	struct rte_mempool *in_mbuf_pool;
	struct rte_mempool *hard_out_mbuf_pool;
	struct rte_mempool *soft_out_mbuf_pool;
	struct rte_mempool *harq_in_mbuf_pool;
	struct rte_mempool *harq_out_mbuf_pool;
} active_devs[RTE_BBDEV_MAX_DEVS];

enum BaseGraph {
    BG1 = 1,
    BG2 = 2
};

static uint8_t nb_active_devs;
// static bool intr_enabled;

// Define constants for the mbuf pool
#define NB_MBUF          8192
#define MBUF_CACHE_SIZE  256
#define RTE_MBUF_DEFAULT_DATAROOM 2048  // This should be adjusted based on your needs

// Create the mbuf pool
// if (mbuf_pool == NULL) {
//     rte_exit(EXIT_FAILURE, "Cannot create mbuf pool: %s\n", rte_strerror(rte_errno));
// }

static constexpr bool kVerbose = false;
static constexpr bool kPrintUplinkInformationBytes = false;
static constexpr float kNoiseLevels[2] = {0.0422, 0.0316};
static constexpr float kSnrLevels[2] = {27.5, 30};
DEFINE_string(profile, "random",
              "The profile of the input user bytes (e.g., 'random', '123')");
DEFINE_string(
    conf_file,
    TOSTRING(PROJECT_DIRECTORY) "/files/config/ci/tddconfig-sim-ul.json",
    "Agora config filename");

float RandFloat(float min, float max) {
  return ((float(rand()) / float(RAND_MAX)) * (max - min)) + min;
}

float RandFloatFromShort(float min, float max) {
  float rand_val =
      ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) *
       (max - min)) +
      min;
  auto rand_val_ushort = static_cast<short>(rand_val * kShrtFltConvFactor);
  rand_val = (float)rand_val_ushort / kShrtFltConvFactor;
  return rand_val;
}

static inline void
set_avail_op(struct active_device *ad, enum rte_bbdev_op_type op_type)
{
	ad->supported_ops |= (1 << op_type);
}

size_t get_varNodes_length(int16_t z, int16_t nRows, int16_t numFillerBits, uint16_t basegraph) {
    size_t length = 0;
    if (basegraph == BG1) {
        length = z * 22 + z * nRows - z * 2 - numFillerBits;
    } else if (basegraph == BG2) {
        length = z * 10 + z * nRows - z * 2 - numFillerBits;
    }
    return length;
}

size_t get_compactedMessageBytes_length(int16_t z, uint16_t basegraph, int16_t numFillerBits) {
    int BG_value = (basegraph == BG1) ? 22 : 10;
    size_t length = z * BG_value - numFillerBits;
    return (length + 7) / 8;  // This performs the ceiling division by 8
}


void configure_bbdev_device(uint16_t dev_id) {
    struct rte_bbdev_info dev_info;
    std::cout << "in configure bbdev" << std::endl;
    uint16_t temp_dev_id = rte_bbdev_find_next(-1);
    uint16_t num_dev = rte_bbdev_count();
    std::cout << "num_dev: " << num_dev << std::endl;
    std::cout << "is valid?" << rte_bbdev_is_valid(temp_dev_id) << std::endl;
    // Get the bbdev device information
    int ret = rte_bbdev_info_get(dev_id, &dev_info);
    if (ret != 0) {
        // Handle error
        return;
    }
    
    // Get the maximum number of queues supported by the device
    uint16_t max_nb_queues = dev_info.num_queues;
    std::cout<<"num queues is: " << unsigned(max_nb_queues) << std::endl;

    // Create the bbdev device
    // rte_vdev_init("baseband_ldpc_sw", NULL);

    // Setup queues for the bbdev device
    std::cout << "no issue here" << std::endl;
    rte_bbdev_setup_queues(dev_id, max_nb_queues, rte_socket_id());

    // Enable interrupts for the bbdev device if supported
    rte_bbdev_intr_enable(dev_id);
}

static int
add_bbdev_dev(uint8_t dev_id, struct rte_bbdev_info *info)
{ 
  std::cout<<"current setting device: " << (unsigned)dev_id << std::endl;
	int ret;
	unsigned int queue_id;
	struct rte_bbdev_queue_conf qconf;
	struct active_device *ad = &active_devs[nb_active_devs];
	unsigned int nb_queues;
	enum rte_bbdev_op_type op_type = RTE_BBDEV_OP_LDPC_DEC;

/* Configure fpga lte fec with PF & VF values
 * if '-i' flag is set and using fpga device
 */

  printf("Configure FPGA 5GNR FEC Driver %s with default values\n",
			info->drv.driver_name);
  // realized that the pf-bbdev should already set it to default, might be needed for future if implemented to Agora. 

	/* Let's refresh this now this is configured */
	rte_bbdev_info_get(dev_id, info);
  std::cout<<"max_num_queues is: " <<  info->drv.max_num_queues  << std::endl;
  std::cout<<"rte lcores is : " << rte_lcore_count() << std::endl;
	nb_queues = RTE_MIN(rte_lcore_count(), info->drv.max_num_queues);
	nb_queues = RTE_MIN(nb_queues, (unsigned int) MAX_QUEUES);

	/* setup device */
  std::cout<<"num queues is: " << unsigned(nb_queues) << std::endl;

	ret = rte_bbdev_setup_queues(dev_id, nb_queues, info->socket_id);
  std::cout<<"!!!!!!!!!!!! ret of setup queue is: " << ret << std::endl;
	if (ret < 0) {
		printf("rte_bbdev_setup_queues(%u, %u, %d) ret %i\n",
				dev_id, nb_queues, info->socket_id, ret);
		return TEST_FAILED;
	}

  ret = rte_bbdev_intr_enable(dev_id);
  std::cout<<"ret for intr enable is: " << ret << std::endl;
	/* configure interrupts if needed */
	// if (intr_enabled) {
	// 	ret = rte_bbdev_intr_enable(dev_id);
	// 	if (ret < 0) {
	// 		printf("rte_bbdev_intr_enable(%u) ret %i\n", dev_id,
	// 				ret);
	// 		return TEST_FAILED;
	// 	}
	// }

	/* setup device queues */
	qconf.socket = info->socket_id;
  std::cout<<"qcof.socket is: " << qconf.socket << std::endl;
	qconf.queue_size = info->drv.default_queue_conf.queue_size;
  std::cout<<"queue size is" << qconf.queue_size << std::endl;
	qconf.priority = 0;
	qconf.deferred_start = 0;
	qconf.op_type = op_type;
  std::cout<<"op type is: " << qconf.op_type << std::endl;

	for (queue_id = 0; queue_id < nb_queues; ++queue_id) {
		ret = rte_bbdev_queue_configure(dev_id, queue_id, &qconf);
		if (ret != 0) {
			printf(
					"Allocated all queues (id=%u) at prio%u on dev%u\n",
					queue_id, qconf.priority, dev_id);
			qconf.priority++;
			ret = rte_bbdev_queue_configure(ad->dev_id, queue_id,
					&qconf);
		}
		if (ret != 0) {
			printf("All queues on dev %u allocated: %u\n",
					dev_id, queue_id);
			break;
		}
		ad->queue_ids[queue_id] = queue_id;
	}
	// TEST_ASSERT(queue_id != 0,
	// 		"ERROR Failed to configure any queues on dev %u",
	// 		dev_id);
	ad->nb_queues = queue_id;
  std::cout<<"ad->nb_queues is: " << unsigned(ad->nb_queues) << std::endl;

	set_avail_op(ad, op_type);

	return TEST_SUCCESS;
}


static int
add_active_device(uint8_t dev_id, struct rte_bbdev_info *info)
{
	int ret;

	active_devs[0].driver_name = info->drv.driver_name;
	active_devs[0].dev_id = dev_id;

	ret = add_bbdev_dev(dev_id, info);
	if (ret == TEST_SUCCESS)
		++nb_active_devs;
	return ret;

}

// static int
// check_dev_cap(const struct rte_bbdev_info *dev_info)
// {
// 	unsigned int i;
// 	unsigned int nb_inputs, nb_soft_outputs, nb_hard_outputs,
// 		nb_harq_inputs, nb_harq_outputs;
// 	const struct rte_bbdev_op_cap *op_cap = dev_info->drv.capabilities;
// 	uint8_t dev_data_endianness = dev_info->drv.data_endianness;

// 	nb_inputs = test_vector.entries[DATA_INPUT].nb_segments;
// 	nb_soft_outputs = test_vector.entries[DATA_SOFT_OUTPUT].nb_segments;
// 	nb_hard_outputs = test_vector.entries[DATA_HARD_OUTPUT].nb_segments;
// 	nb_harq_inputs  = test_vector.entries[DATA_HARQ_INPUT].nb_segments;
// 	nb_harq_outputs = test_vector.entries[DATA_HARQ_OUTPUT].nb_segments;

// 	for (i = 0; op_cap->type != RTE_BBDEV_OP_NONE; ++i, ++op_cap) {
// 		if (op_cap->type != test_vector.op_type)
// 			continue;

// 		if (dev_data_endianness == RTE_BIG_ENDIAN)
// 			convert_op_data_to_be();

// 		if (op_cap->type == RTE_BBDEV_OP_TURBO_DEC) {
// 			const struct rte_bbdev_op_cap_turbo_dec *cap =
// 					&op_cap->cap.turbo_dec;
// 			/* Ignore lack of soft output capability, just skip
// 			 * checking if soft output is valid.
// 			 */
// 			if ((test_vector.turbo_dec.op_flags &
// 					RTE_BBDEV_TURBO_SOFT_OUTPUT) &&
// 					!(cap->capability_flags &
// 					RTE_BBDEV_TURBO_SOFT_OUTPUT)) {
// 				printf(
// 					"INFO: Device \"%s\" does not support soft output - soft output flags will be ignored.\n",
// 					dev_info->dev_name);
// 				clear_soft_out_cap(
// 					&test_vector.turbo_dec.op_flags);
// 			}

// 			if (!flags_match(test_vector.turbo_dec.op_flags,
// 					cap->capability_flags))
// 				return TEST_FAILED;
// 			if (nb_inputs > cap->num_buffers_src) {
// 				printf("Too many inputs defined: %u, max: %u\n",
// 					nb_inputs, cap->num_buffers_src);
// 				return TEST_FAILED;
// 			}
// 			if (nb_soft_outputs > cap->num_buffers_soft_out &&
// 					(test_vector.turbo_dec.op_flags &
// 					RTE_BBDEV_TURBO_SOFT_OUTPUT)) {
// 				printf(
// 					"Too many soft outputs defined: %u, max: %u\n",
// 						nb_soft_outputs,
// 						cap->num_buffers_soft_out);
// 				return TEST_FAILED;
// 			}
// 			if (nb_hard_outputs > cap->num_buffers_hard_out) {
// 				printf(
// 					"Too many hard outputs defined: %u, max: %u\n",
// 						nb_hard_outputs,
// 						cap->num_buffers_hard_out);
// 				return TEST_FAILED;
// 			}
// 			if (intr_enabled && !(cap->capability_flags &
// 					RTE_BBDEV_TURBO_DEC_INTERRUPTS)) {
// 				printf(
// 					"Dequeue interrupts are not supported!\n");
// 				return TEST_FAILED;
// 			}

// 			return TEST_SUCCESS;
// 		} else if (op_cap->type == RTE_BBDEV_OP_TURBO_ENC) {
// 			const struct rte_bbdev_op_cap_turbo_enc *cap =
// 					&op_cap->cap.turbo_enc;

// 			if (!flags_match(test_vector.turbo_enc.op_flags,
// 					cap->capability_flags))
// 				return TEST_FAILED;
// 			if (nb_inputs > cap->num_buffers_src) {
// 				printf("Too many inputs defined: %u, max: %u\n",
// 					nb_inputs, cap->num_buffers_src);
// 				return TEST_FAILED;
// 			}
// 			if (nb_hard_outputs > cap->num_buffers_dst) {
// 				printf(
// 					"Too many hard outputs defined: %u, max: %u\n",
// 					nb_hard_outputs, cap->num_buffers_dst);
// 				return TEST_FAILED;
// 			}
// 			if (intr_enabled && !(cap->capability_flags &
// 					RTE_BBDEV_TURBO_ENC_INTERRUPTS)) {
// 				printf(
// 					"Dequeue interrupts are not supported!\n");
// 				return TEST_FAILED;
// 			}

// 			return TEST_SUCCESS;
// 		} else if (op_cap->type == RTE_BBDEV_OP_LDPC_ENC) {
// 			const struct rte_bbdev_op_cap_ldpc_enc *cap =
// 					&op_cap->cap.ldpc_enc;

// 			if (!flags_match(test_vector.ldpc_enc.op_flags,
// 					cap->capability_flags)){
// 				printf("Flag Mismatch\n");
// 				return TEST_FAILED;
// 			}
// 			if (nb_inputs > cap->num_buffers_src) {
// 				printf("Too many inputs defined: %u, max: %u\n",
// 					nb_inputs, cap->num_buffers_src);
// 				return TEST_FAILED;
// 			}
// 			if (nb_hard_outputs > cap->num_buffers_dst) {
// 				printf(
// 					"Too many hard outputs defined: %u, max: %u\n",
// 					nb_hard_outputs, cap->num_buffers_dst);
// 				return TEST_FAILED;
// 			}
// 			if (intr_enabled && !(cap->capability_flags &
// 					RTE_BBDEV_LDPC_ENC_INTERRUPTS)) {
// 				printf(
// 					"Dequeue interrupts are not supported!\n");
// 				return TEST_FAILED;
// 			}

// 			return TEST_SUCCESS;
// 		} else if (op_cap->type == RTE_BBDEV_OP_LDPC_DEC) {
// 			const struct rte_bbdev_op_cap_ldpc_dec *cap =
// 					&op_cap->cap.ldpc_dec;

// 			if (!flags_match(test_vector.ldpc_dec.op_flags,
// 					cap->capability_flags)){
// 				printf("Flag Mismatch\n");
// 				return TEST_FAILED;
// 			}
// 			if (nb_inputs > cap->num_buffers_src) {
// 				printf("Too many inputs defined: %u, max: %u\n",
// 					nb_inputs, cap->num_buffers_src);
// 				return TEST_FAILED;
// 			}
// 			if (nb_hard_outputs > cap->num_buffers_hard_out) {
// 				printf(
// 					"Too many hard outputs defined: %u, max: %u\n",
// 					nb_hard_outputs,
// 					cap->num_buffers_hard_out);
// 				return TEST_FAILED;
// 			}
// 			if (nb_harq_inputs > cap->num_buffers_hard_out) {
// 				printf(
// 					"Too many HARQ inputs defined: %u, max: %u\n",
// 					nb_harq_inputs,
// 					cap->num_buffers_hard_out);
// 				return TEST_FAILED;
// 			}
// 			if (nb_harq_outputs > cap->num_buffers_hard_out) {
// 				printf(
// 					"Too many HARQ outputs defined: %u, max: %u\n",
// 					nb_harq_outputs,
// 					cap->num_buffers_hard_out);
// 				return TEST_FAILED;
// 			}
// 			if (intr_enabled && !(cap->capability_flags &
// 					RTE_BBDEV_LDPC_DEC_INTERRUPTS)) {
// 				printf(
// 					"Dequeue interrupts are not supported!\n");
// 				return TEST_FAILED;
// 			}
// 			if (intr_enabled && (test_vector.ldpc_dec.op_flags &
// 				(RTE_BBDEV_LDPC_HQ_COMBINE_IN_ENABLE |
// 				RTE_BBDEV_LDPC_HQ_COMBINE_OUT_ENABLE |
// 				RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_LOOPBACK
// 					))) {
// 				printf("Skip loop-back with interrupt\n");
// 				return TEST_FAILED;
// 			}
// 			return TEST_SUCCESS;
// 		}
// 	}

// 	if ((i == 0) && (test_vector.op_type == RTE_BBDEV_OP_NONE))
// 		return TEST_SUCCESS; /* Special case for NULL device */

// 	return TEST_FAILED;
// }


int main(int argc, char* argv[]) {
  std::string core_list = std::to_string(34) + "," + std::to_string(35) + "," + std::to_string(36) + "," + std::to_string(37);
  const char* rte_argv[] = {"txrx",        "-l",           core_list.c_str(),
                            "--log-level", "lib.eal:info", nullptr};
  int rte_argc = static_cast<int>(sizeof(rte_argv) / sizeof(rte_argv[0])) - 1;

  // Initialize DPDK environment
  std::cout<<"getting ready to init dpdk" << std::endl;
  int ret = rte_eal_init(rte_argc, const_cast<char**>(rte_argv));
  RtAssert(
      ret >= 0,
      "Failed to initialize DPDK.  Are you running with root permissions?");

  // int ret = rte_eal_init(rte_argc, const_cast<char**>(rte_argv));
  // RtAssert(
  //     ret >= 0,
  //     "Failed to initialize DPDK.  Are you running with root permissions?");
  // std::printf("%s initialized\n", rte_version());

  std::cout<<"trying to setup HW acc100" << std::endl;
  int ret_acc;
  uint8_t dev_id;
  // uint8_t nb_devs_added = 0;
  struct rte_bbdev_info info;
  // std::cout << "dev_id: " << unsigned(dev_id) << std::endl;
  // RTE_BBDEV_FOREACH(dev_id) {
  //   std::cout << "dev_id: " << unsigned(dev_id) << std::endl;
  rte_bbdev_info_get(dev_id, &info);

  // if (check_dev_cap(&info)) {
  //   printf(
  //     "Device %d (%s) does not support specified capabilities\n",
  //       dev_id, info.dev_name);
  //   continue;
  // }
  // if (check_dev_cap(&info)) {
  //   printf(
  //     "Device %d (%s) does not support specified capabilities\n",
  //       dev_id, info.dev_name);
  //   continue;
  // }

  const struct rte_bbdev_info *dev_info = &info;
  const struct rte_bbdev_op_cap *op_cap = dev_info->drv.capabilities;
  for (unsigned int i = 0; op_cap->type != RTE_BBDEV_OP_NONE; ++i, ++op_cap) {
    std::cout<<"capabilities is: " << op_cap->type << std::endl;
  }

  
  ret_acc = add_active_device(dev_id, &info);
  std::cout << "ret: " << ret_acc << std::endl;
  if (ret_acc != 0) {
    printf("Adding active bbdev %s skipped\n",
        info.dev_name);
    // continue;
  }
      // nb_devs_added++;
	// }
  // std::cout << "nb_devs_added: " << unsigned(nb_devs_added) << std::endl;

  std::cout<<"[1] added device name is: " << info.dev_name << std::endl;
  std::cout<<"[1] added device socket id is: " << info.socket_id << std::endl;
  std::cout<<"[1] added device driver name is: " << info.drv.driver_name << std::endl;
  std::cout<<"[1] Number of queues currently configured is: " << info.num_queues << std::endl;

  rte_bbdev_intr_enable(dev_id);


  struct active_device *ad;
  ad = &active_devs[dev_id];
  rte_bbdev_info_get(ad->dev_id, &info);

  std::cout<<"double check: !!!!" << std::endl;
  std::cout<<"[2] added device name is: " << info.dev_name << std::endl;
  std::cout<<"[2] added device socket id is: " << info.socket_id << std::endl;
  std::cout<<"[2] added device driver name is: " << info.drv.driver_name << std::endl;
  std::cout<<"[2] Number of queues currently configured is: " << info.num_queues << std::endl;
  // first check if deivce has started:
  TEST_ASSERT_SUCCESS(rte_bbdev_stats_reset(dev_id),
				"Failed to reset stats of bbdev %u", dev_id);

  TEST_ASSERT_SUCCESS(rte_bbdev_start(dev_id),
				"Failed to reset stats of bbdev %u", dev_id);
  // return value 0 means success
  // std::cout<<"status of device is: " << rte_bbdev_start(dev_id) << std::endl;
  
  // ut_teardown test
  // struct rte_bbdev_stats stats;
  // rte_bbdev_stats_get(dev_id, &stats);

		/* Stop the device */
	// rte_bbdev_stop(dev_id);

  // rte_bbdev_info_get(ad->dev_id, &info);

  // std::cout<<"third check: !!!!" << std::endl;
  // std::cout<<"[3] added device name is: " << info.dev_name << std::endl;
  // std::cout<<"[3] added device socket id is: " << info.socket_id << std::endl;
  // std::cout<<"[3] added device driver name is: " << info.drv.driver_name << std::endl;
  // std::cout<<"[3] Number of queues currently configured is: " << info.num_queues << std::endl;

  // shoudl create mempool for bbdev operations
  uint16_t num_ops = 2047;
  uint16_t burst_sz = 1;
  // int rte_alloc;
  struct rte_mempool *ops_mp;
  struct rte_mempool* in_mbuf_pool;
  struct rte_mempool* out_mbuf_pool;
  // struct rte_mempool *in_mp;
  // struct rte_mempool *out_mp;
  ops_mp = rte_bbdev_op_pool_create("RTE_BBDEV_OP_LDPC_DEC_poo", RTE_BBDEV_OP_LDPC_DEC,
      num_ops, OPS_CACHE_SIZE, rte_socket_id());
  std::cout<<"num ops is: " << unsigned(num_ops) << std::endl;
  std::cout<<"socket id is: " << unsigned(rte_socket_id()) << std::endl;

  in_mbuf_pool = rte_pktmbuf_pool_create("IN_MBUF_POOL", 2047, 0, 0, 22744, 0);
  out_mbuf_pool = rte_pktmbuf_pool_create("OUT_MBUF_POOL", 2047, 0, 0, 22744, 0);

  if (in_mbuf_pool == nullptr or out_mbuf_pool == nullptr) {
    std::cerr << "Error: Unable to create mbuf pool: " << rte_strerror(rte_errno) << std::endl;
    return -1;  // Exit the program with an error code
  }
  // debugging message to ensure mempool and allocation:
  std::cout << "burst_sz: " << burst_sz << std::endl;
  auto socket_id = rte_socket_id();
  std::cout << "Socket ID: " << socket_id << std::endl;

  if (ops_mp == nullptr) {
    std::cerr << "Error: Failed to create memory pool for bbdev operations." << std::endl;
  } else {
      std::cout << "Memory pool for bbdev operations created successfully." << std::endl;
  }

  // struct rte_bbdev_dec_op* dec_ops[1];
  // uint16_t num_ops = 1;
  // uint16_t dev_id = 0;  // Assuming bbdev device ID is 0
  // uint16_t queue_id = 0;  // Assuming queue ID is 0

  // Allocate memory for bbdev operations
  struct rte_bbdev_dec_op *ops_enq[MAX_PKT_BURST], *ops_deq[MAX_PKT_BURST];


  int rte_alloc = rte_bbdev_dec_op_alloc_bulk(ops_mp, ops_enq, burst_sz);

  // check capability:
  const struct rte_bbdev_op_cap *cap = info.drv.capabilities;
  rte_bbdev_info_get(dev_id, &info);
  for (unsigned int i = 0; cap->type != RTE_BBDEV_OP_NONE; ++i, ++cap) {
    std::cout<<"cap is: " << cap->type  << std::endl;
  }
  // std::cout << "cap is: " << cap->type << std::endl;
  // output here cap is:1, but 1 correspond to RTE_BBDEV_OP_TURBO_DEC,
  // the perf gives the cap type to be 3

  // may need to create reference ops 


  std::cout<<"rte_alloc is: " << rte_alloc << std::endl;
  RtAssert(
      rte_alloc >= 0,
      "Failed to alloc dec operation.\n");

  //debug print
  std::cout << "rte_alloc value: " << rte_alloc << std::endl;
  if (rte_alloc < 0) {
      std::cerr << "Error: Failed to allocate bulk operations for bbdev." << std::endl;
  }

  // init test op paramters - seen above

  // create reference ldpc_dec operation
  
  // fill queue buffer - in each decode task

  //free buffers

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(0.0, 1.0);

  const std::string cur_directory = TOSTRING(PROJECT_DIRECTORY);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto cfg = std::make_unique<Config>(FLAGS_conf_file.c_str());
  Direction dir =
      cfg->Frame().NumULSyms() > 0 ? Direction::kUplink : Direction::kDownlink;

  const DataGenerator::Profile profile =
      FLAGS_profile == "123" ? DataGenerator::Profile::kProfile123
                             : DataGenerator::Profile::kRandom;
  DataGenerator data_generator(cfg.get(), 0 /* RNG seed */, profile);

  std::printf(
      "DataGenerator: Config file: %s, data profile = %s\n",
      FLAGS_conf_file.c_str(),
      profile == DataGenerator::Profile::kProfile123 ? "123" : "random");

  std::printf("DataGenerator: Using %s-orthogonal pilots\n",
              cfg->FreqOrthogonalPilot() ? "frequency" : "time");

  std::printf("DataGenerator: Generating encoded and modulated data\n");
  srand(time(nullptr));

  // Step 1: Generate the information buffers and LDPC-encoded buffers for
  // uplink
  size_t num_symbols_per_cb = 1;
  size_t bits_per_symbol = cfg->OfdmDataNum() * cfg->ModOrderBits(dir);
  if (cfg->LdpcConfig(dir).NumCbCodewLen() > bits_per_symbol) {
    num_symbols_per_cb =
        (cfg->LdpcConfig(dir).NumCbCodewLen() + bits_per_symbol - 1) /
        bits_per_symbol;
  }
  size_t num_cbs_per_ue = cfg->Frame().NumDataSyms() / num_symbols_per_cb;
  std::printf("Number of symbols per block: %zu, blocks per frame: %zu\n",
              num_symbols_per_cb, num_cbs_per_ue);

  const size_t num_codeblocks = num_cbs_per_ue * cfg->UeAntNum();
  std::printf("Total number of blocks: %zu\n", num_codeblocks);
  size_t input_size = Roundup<64>(
      LdpcEncodingInputBufSize(cfg->LdpcConfig(dir).BaseGraph(),
                               cfg->LdpcConfig(dir).ExpansionFactor()));
  auto* input_ptr = new int8_t[input_size];
  for (size_t noise_id = 0; noise_id < 2; noise_id++) {
    std::vector<std::vector<int8_t>> information(num_codeblocks);
    std::vector<std::vector<int8_t>> encoded_codewords(num_codeblocks);
    for (size_t i = 0; i < num_codeblocks; i++) {
      data_generator.GenRawData(dir, information.at(i),
                                i % cfg->UeAntNum() /* UE ID */);
      std::memcpy(input_ptr, information.at(i).data(), input_size);
      data_generator.GenCodeblock(dir, input_ptr, encoded_codewords.at(i));
    }

    // Save uplink information bytes to file
    const size_t input_bytes_per_cb =
        BitsToBytes(LdpcNumInputBits(cfg->LdpcConfig(dir).BaseGraph(),
                                     cfg->LdpcConfig(dir).ExpansionFactor()));
    if (kPrintUplinkInformationBytes) {
      std::printf("Uplink information bytes\n");
      for (size_t n = 0; n < num_codeblocks; n++) {
        std::printf("Symbol %zu, UE %zu\n", n / cfg->UeAntNum(),
                    n % cfg->UeAntNum());
        for (size_t i = 0; i < input_bytes_per_cb; i++) {
          std::printf("%u ", (uint8_t)information[n][i]);
        }
        std::printf("\n");
      }
    }

    Table<complex_float> modulated_codewords;
    modulated_codewords.Calloc(num_codeblocks, cfg->OfdmDataNum(),
                               Agora_memory::Alignment_t::kAlign64);
    Table<int8_t> demod_data_all_symbols;
    demod_data_all_symbols.Calloc(num_codeblocks, cfg->OfdmDataNum() * 8,
                                  Agora_memory::Alignment_t::kAlign64);
    std::vector<uint8_t> mod_input(cfg->OfdmDataNum());

    // Modulate, add noise, and demodulate the encoded codewords
    for (size_t i = 0; i < num_codeblocks; i++) {
      AdaptBitsForMod(
          reinterpret_cast<const uint8_t*>(&encoded_codewords[i][0]),
          &mod_input[0], cfg->LdpcConfig(dir).NumEncodedBytes(),
          cfg->ModOrderBits(dir));

      for (size_t j = 0; j < cfg->OfdmDataNum(); j++) {
        modulated_codewords[i][j] =
            ModSingleUint8(mod_input[j], cfg->ModTable(dir));
      }

      for (size_t j = 0; j < cfg->OfdmDataNum(); j++) {
        complex_float noise = {static_cast<float>(distribution(generator)) *
                                   kNoiseLevels[noise_id],
                               static_cast<float>(distribution(generator)) *
                                   kNoiseLevels[noise_id]};
        modulated_codewords[i][j].re = modulated_codewords[i][j].re + noise.re;
        modulated_codewords[i][j].im = modulated_codewords[i][j].im + noise.im;
      }

      switch (cfg->ModOrderBits(dir)) {
        case (4):
          Demod16qamSoftAvx2((float*)modulated_codewords[i],
                             demod_data_all_symbols[i], cfg->OfdmDataNum());
          break;
        case (6):
          Demod64qamSoftAvx2((float*)modulated_codewords[i],
                             demod_data_all_symbols[i], cfg->OfdmDataNum());
          break;
        default:
          std::printf("Demodulation: modulation type %s not supported!\n",
                      cfg->Modulation(dir).c_str());
      }
    }

    const LDPCconfig& ldpc_config = cfg->LdpcConfig(dir);

// own ldpc method to decode
    // uint16_t dev_id = std::stoi("0000:17:00.0", nullptr, 16);
    // configure_bbdev_device(dev_id);
    // configure_bbdev_device(0d5c);
    for (int i = 0; i < burst_sz; ++i) {

      ops_enq[i]->ldpc_dec.basegraph = ldpc_config.BaseGraph();
      ops_enq[i]->ldpc_dec.z_c = ldpc_config.ExpansionFactor();
      ops_enq[i]->ldpc_dec.n_filler = 0;
      ops_enq[i]->ldpc_dec.iter_max = ldpc_config.MaxDecoderIter();

      ops_enq[i]->status = 0; // Default value
      ops_enq[i]->mempool = ops_mp; 
      ops_enq[i]->opaque_data = nullptr; // U
      // bbdev_ops_burst[i]->ldpc_dec.input.data = (struct rte_mbuf*)input_ptr;

      ops_enq[i]->ldpc_dec.op_flags = RTE_BBDEV_LDPC_HQ_COMBINE_IN_ENABLE | RTE_BBDEV_LDPC_CRC_24A_ATTACH;
      ops_enq[i]->ldpc_dec.rv_index = 0;
      ops_enq[i]->ldpc_dec.n_cb = ldpc_config.NumCbCodewLen();
      ops_enq[i]->ldpc_dec.q_m = 4;
      ops_enq[i]->ldpc_dec.code_block_mode = 1;
    }

    std::cout<<"general ldpc setting complete" << std::endl;

    struct bblib_ldpc_decoder_5gnr_request ldpc_decoder_5gnr_request {};
    struct bblib_ldpc_decoder_5gnr_response ldpc_decoder_5gnr_response {};

    // Decoder setup
    // ldpc_decoder_5gnr_request.numChannelLlrs = ldpc_config.NumCbCodewLen();
    // ldpc_decoder_5gnr_request.numFillerBits = 0;                                  // added
    // ldpc_decoder_5gnr_request.maxIterations = ldpc_config.MaxDecoderIter();       // added
    // ldpc_decoder_5gnr_request.enableEarlyTermination =
    //     ldpc_config.EarlyTermination();
    // ldpc_decoder_5gnr_request.Zc = ldpc_config.ExpansionFactor();                 // added
    // ldpc_decoder_5gnr_request.baseGraph = ldpc_config.BaseGraph();                // added
    // ldpc_decoder_5gnr_request.nRows = ldpc_config.NumRows();
    
    // ldpc_decoder_5gnr_response.numMsgBits = ldpc_config.NumCbLen();
    auto* resp_var_nodes = static_cast<int16_t*>(
        Agora_memory::PaddedAlignedAlloc(Agora_memory::Alignment_t::kAlign64,
                                         1024 * 1024 * sizeof(int16_t)));
    ldpc_decoder_5gnr_response.varNodes = resp_var_nodes;

    Table<uint8_t> decoded_codewords;
    decoded_codewords.Calloc(num_codeblocks, cfg->OfdmDataNum(),
                             Agora_memory::Alignment_t::kAlign64);

    double freq_ghz = GetTime::MeasureRdtscFreq();
    size_t start_tsc = GetTime::WorkerRdtsc();
    std::cout<<"num_codeblocks is: " << num_codeblocks << std::endl;
    for (size_t i = 0; i < num_codeblocks; i++) {
      size_t varNodes_length = get_varNodes_length(ldpc_config.ExpansionFactor(), ldpc_config.NumRows(), 0, ldpc_config.BaseGraph());
      size_t compactedMessageBytes_length = get_compactedMessageBytes_length(ldpc_config.ExpansionFactor(), ldpc_config.BaseGraph(), 0);
      std::cout<<"varNodes_length is: " << varNodes_length << std::endl;
      std::cout<<"compacted message byts length is " << compactedMessageBytes_length << std::endl;

      struct rte_mbuf* mbuf = rte_pktmbuf_alloc(in_mbuf_pool);
      // std::cout<<"no issue after mubf" << std::endl;
      if (!mbuf) {
          std::cerr << "Failed to allocate mbuf." << std::endl;
          continue;
      }

      struct rte_mbuf* mbuf_decoded = rte_pktmbuf_alloc(out_mbuf_pool);
      if (!mbuf_decoded) {
          std::cerr << "Failed to allocate mbuf for decoded data." << std::endl;
          // Handle the error, e.g., continue to the next iteration or exit the loop
          continue;
      }

      // Assuming demod_data_all_symbols[i] is a pointer to the actual data and not another mbuf
      char* data_ptr = rte_pktmbuf_mtod(mbuf, char*);
      std::memcpy(data_ptr, demod_data_all_symbols[i], varNodes_length);

      if (!data_ptr) {
        std::cerr << "Failed to append data to mbuf." << std::endl;
        rte_pktmbuf_free(mbuf);
        continue;
        // Handle the error
      }
      std::cout<<"data ptr finished copy" << std::endl;

      ops_enq[0]->ldpc_dec.input.data = mbuf;
      std::cout<<"data being populated to ldpc_dec_input" << std::endl;

      char* decoded_data_ptr = rte_pktmbuf_mtod(mbuf_decoded, char*);
      std::memcpy(decoded_data_ptr, decoded_codewords[i], compactedMessageBytes_length);

      if (!decoded_data_ptr) {
          std::cerr << "Failed to append decoded data to mbuf." << std::endl;
          rte_pktmbuf_free(mbuf_decoded);
          // Handle the error, e.g., continue to the next iteration or exit the loop
          continue;
      }

      // bbdev_ops_burst[0]->ldpc_dec.input.data = (struct rte_mbuf*)demod_data_all_symbols[i];
      // bbdev_ops_burst[i]->ldpc_dec.input.data = reinterpret_cast<void*>(demod_data_all_symbols[i]);
      ops_enq[0]->ldpc_dec.input.length = varNodes_length;
      std::cout<<"data length being populated to ldpc_dec_input" << std::endl;

      ldpc_decoder_5gnr_request.varNodes = demod_data_all_symbols[i];
      ldpc_decoder_5gnr_response.compactedMessageBytes = decoded_codewords[i];

      ops_enq[0]->ldpc_dec.hard_output.data = mbuf_decoded;
      ops_enq[0]->ldpc_dec.hard_output.length = compactedMessageBytes_length;  // Set the correct length
      // bbdev_ops_burst[0]->ldpc_dec.hard_output.data = (struct rte_mbuf*)decoded_codewords[i];

      
      uint16_t enq = 0, deq = 0;
      bool first_time = true;
      uint64_t start_time = 0, last_time = 0;

      // start_time = rte_rdtsc_precise();
      int j = 0;
			ops_enq[0]->opaque_data = (void *)(uintptr_t)j;
      std::cout<<"start to enqueue" << std::endl;
      enq = rte_bbdev_enqueue_ldpc_dec_ops(0, 0, &ops_enq[0], 1);
      std::cout<<"enq is: " << (unsigned)enq << std::endl;
      std::cout<<"rte operation enqueue done once!" << std::endl;

      int max_retries = 10;
      int retry_count = 0;

      while (deq == 0 && retry_count < max_retries) {
          rte_delay_ms(10);  // Wait for 10 milliseconds
          deq = rte_bbdev_dequeue_ldpc_dec_ops(0, 0, &ops_deq[0], 1);
          retry_count++;
      }

      if (deq == 0) {
          std::cerr << "Failed to dequeue after " << max_retries << " attempts." << std::endl;
          // Handle the error
      } else {
          std::cout << "Dequeue successful after " << retry_count << " attempts." << std::endl;
      }

      // do {
      // deq = rte_bbdev_dequeue_ldpc_dec_ops(0, 0, &bbdev_ops_burst[deq], 1);
      std::cout<<"deq is: " << (unsigned)deq << std::endl;
      std::cout<<"rte operation dequeue done once!" << std::endl;
		  // } while (unlikely(burst_sz != deq));

      // bblib_ldpc_decoder_5gnr(&ldpc_decoder_5gnr_request,
      //                         &ldpc_decoder_5gnr_response);
      rte_pktmbuf_free(mbuf);
      
    }

    size_t duration = GetTime::WorkerRdtsc() - start_tsc;
    std::printf("Decoding of %zu blocks takes %.2f us per block\n",
                num_codeblocks,
                GetTime::CyclesToUs(duration, freq_ghz) / num_codeblocks);

    // Correctness check
    size_t error_num = 0;
    size_t total = num_codeblocks * ldpc_config.NumCbLen();
    size_t block_error_num = 0;

    for (size_t i = 0; i < num_codeblocks; i++) {
      size_t error_in_block = 0;
      for (size_t j = 0; j < ldpc_config.NumCbLen() / 8; j++) {
        auto input = static_cast<uint8_t>(information.at(i).at(j));
        uint8_t output = decoded_codewords[i][j];
        if (input != output) {
          for (size_t k = 0; k < 8; k++) {
            uint8_t mask = 1 << k;
            if ((input & mask) != (output & mask)) {
              error_num++;
              error_in_block++;
            }
          }
          // std::printf("block %zu j: %zu: (%u, %u)\n", i, j,
          //     (uint8_t)information[i][j], decoded_codewords[i][j]);
        }
      }
      if (error_in_block > 0) {
        block_error_num++;
        // std::printf("errors in block %zu: %zu\n", i, error_in_block);
      }
    }

    std::printf(
        "Noise: %.3f, snr: %.1f dB, error rate: %zu/%zu = %.6f, block "
        "error: "
        "%zu/%zu = %.6f\n",
        kNoiseLevels[noise_id], kSnrLevels[noise_id], error_num, total,
        1.f * error_num / total, block_error_num, num_codeblocks,
        1.f * block_error_num / num_codeblocks);

    std::free(resp_var_nodes);
    modulated_codewords.Free();
    demod_data_all_symbols.Free();
    decoded_codewords.Free();
  }
  delete[] input_ptr;
  return 0;
}
