/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-02-05     Jianjia Ma   The first version
 * 2019-02-10     Jianjia Ma   Compiler supports dense net connection
 */

#ifndef __NNOM_H__
#define __NNOM_H__

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include <math.h>

#include "nnom_port.h"

#define NNOM_ALIGN  (sizeof(char*))     // alignment when doing memory ops. Equal to size of pointer in byte.
#define q7_t 	int8_t
#define q15_t 	int16_t
#define q31_t 	int32_t
#define q63_t 	int64_t

/* version */
#define NNOM_MAJORVERSION     0              /**< major version number */
#define NNOM_SUBVERSION       4              /**< minor version number */
#define NNOM_REVISION         3              /**< revise version number */
#define NNOM_VERSION          ((NNOM_MAJORVERSION * 10000) + (NNOM_SUBVERSION * 100) + NNOM_REVISION)

#ifdef ARM_NN_TRUNCATE
#define NNOM_TRUNCATE
#endif

#ifndef NNOM_TRUNCATE 
    #define NNOM_ROUND(out_shift) ((0x1 << out_shift) >> 1 )
#else
    #define NNOM_ROUND(out_shift) 0
#endif
										 
typedef enum
{
	NN_SUCCESS = 0,			/**< No error */
	NN_ARGUMENT_ERROR = -1, /**< One or more arguments are incorrect */
	NN_LENGTH_ERROR = -2,   /**< Length of data buffer is incorrect */
	NN_SIZE_MISMATCH = -3,  /**< Size of matrices is not compatible with the operation. */
	NN_NANINF = -4,			/**< Not-a-number (NaN) or infinity is generated */
	NN_SINGULAR = -5,		/**< Generated by matrix inversion if the input matrix is singular and cannot be inverted. */
	NN_TEST_FAILURE = -6,   /**< Test Failed  */
	NN_NO_MEMORY = -7,
	NN_MORE_TODO = -8
} nnom_status_t;

typedef enum
{
	NNOM_INVALID = 0,
	NNOM_BASE,
	NNOM_INPUT,
	NNOM_OUTPUT,
	NNOM_CONV_2D,
	NNOM_DW_CONV_2D,
	NNOM_CONV2D_TRANS,
	NNOM_BATCHNORM,
	NNOM_DENSE,
	NNOM_ZERO_PADDING,
	NNOM_CROPPING,
	NNOM_RNN,
	NNOM_ACTIVATION,
	NNOM_RELU,
	NNOM_LEAKY_RELU,
	NNOM_ADV_RELU,
	NNOM_SIGMOID,
	NNOM_TANH,
	NNOM_SOFTMAX,
	NNOM_MAXPOOL,
	NNOM_GLOBAL_MAXPOOL,
	NNOM_AVGPOOL,
	NNOM_GLOBAL_AVGPOOL,
	NNOM_SUMPOOL,
	NNOM_GLOBAL_SUMPOOL,
	NNOM_UPSAMPLE,
	NNOM_FLATTEN,
    NNOM_RESHAPE,
	NNOM_LAMBDA,
	NNOM_CONCAT,
	NNOM_ADD,
	NNOM_SUB,
	NNOM_MULT,
	NNOM_TYPE_MAX

} nnom_layer_type_t;

#define DEFUALT_LAYER_NAMES \
	{                       \
		"Unknown",          \
			"Base",			\
			"Input",        \
			"Output",       \
			"Conv2D",       \
			"DW_Conv2D",    \
			"Conv2DTrsp",    \
			"BatchNorm",	\
			"Dense",        \
			"ZeroPad",	    \
			"Cropping",     \
			"RNN",          \
			"Activation",   \
			"ReLU",         \
			"Leaky_ReLU",	\
			"Adv_ReLU",	    \
			"Sigmoid",      \
			"Tanh",         \
			"Softmax",      \
			"MaxPool",      \
			"GL_MaxPool",	\
			"AvgPool",      \
			"GL_AvgPool",	\
			"SumPool",		\
			"GL_SumPool",	\
			"UpSample",		\
			"Flatten",      \
            "Reshape",      \
			"Lambda",       \
			"Concat",       \
			"Add",          \
			"Sub",          \
			"Mult",         \
	}
extern const char default_layer_names[][12];

// We dont count softmax an activation here, softmax is instanced as a layer
typedef enum
{
    ACT_UNKNOWN = 0,
	ACT_RELU,
	ACT_LEAKY_RELU,
	ACT_ADV_RELU,
	ACT_TANH,
	ACT_SIGMOID,
    ACT_HARD_TANH,
    ACT_HARD_SIGMOID
} nnom_activation_type_t;

#define ACTIVATION_NAMES \
	{                    \
        "Unknown",          \
		"ReLU",          \
		"LkyReLU",		 \
		"AdvReLU",		\
		"TanH",      \
		"Sigmoid",   \
        "HrdTanH",      \
		"HrdSigd",   \
	}
extern const char default_activation_names[][8];

// RNN cell type
typedef enum
{
	NNOM_UNKOWN_CELL = 0,
	NNOM_SIMPLE_CELL,
	NNOM_GRU_CELL,
	NNOM_LSTM_CELL,
	NNOM_CELL_TYPE_MAX
} nnom_rnn_cell_type_t;

#define DEFUALT_CELL_NAMES \
	{                    \
		"Unknown",          \
		"Simple",		 \
		"GRU",		\
		"LSTM",      \
	}
extern const char default_cell_names[][8];


// parameters
typedef enum
{
	PADDING_VALID = 0,
	PADDING_SAME
} nnom_padding_t;

#define NNOM_TENSOR_BUF_NULL     (0)	// This buffer is not in used
#define NNOM_TENSOR_BUF_TEMP     (1)  // The memory in IO is temporary occupided, can be reused by other layer once the computation is done.
#define NNOM_TENSOR_BUF_RESERVED (2)  // the mem is reserve for this layer only (not to be reused by other layer.

// currently used in compiling.
#define NNOM_BUF_EMPTY   (0)
#define NNOM_BUF_FILLED  (1)

// basic types
#define nnom_qformat_param_t int32_t // this should match the backend, need a better way to do it. 
#define nnom_shape_data_t uint16_t

typedef struct _nnom_3d_shape_t
{
	nnom_shape_data_t h, w, c;
} nnom_3d_shape_t;

typedef struct _nnom_border_t
{
	nnom_shape_data_t top, bottom, left, right;
} nnom_border_t;

// nnom_3d_shape_axis_t type provide the axis[] format access to nnom_3d_shape_t
typedef union {
	nnom_3d_shape_t s;
	nnom_shape_data_t axis[sizeof(nnom_3d_shape_t) / sizeof(nnom_shape_data_t)];
} nnom_3d_shape_axis_t;

// tensor quantisation types
typedef enum
{
	NNOM_QTYPE_PER_TENSOR = 0,
	NNOM_QTYPE_PER_AXIS = 1
} nnom_qtype_t;

typedef struct _nnom_weights
{
	const void *p_value;
	nnom_qformat_param_t shift;
} nnom_weight_t;

typedef struct _nnom_bias
{
	const void *p_value;
	nnom_qformat_param_t shift;
} nnom_bias_t;

// experimental                   
typedef struct _nnom_tensor_t
{
	void* p_data;			// value
	nnom_shape_data_t *dim; // dimension of this tensor
	nnom_qformat_param_t *q_dec;	// number of decimal bit for Q format (scale)
	nnom_qformat_param_t *q_offset;	// offset for each channel
	nnom_qtype_t qtype;			// the quantisation type	
	uint8_t num_dim;			// the number of dimension
	uint8_t bitwidth;			// the data bit width, only support 8bit now
} nnom_tensor_t;

// nn wrappers
typedef struct _nnom_layer_t 	nnom_layer_t;
typedef struct _nnom_layer_io_t nnom_layer_io_t;
typedef struct _nnom_layer_hook_t nnom_layer_hook_t;
typedef struct _nnom_mem_block_t nnom_mem_block_t;

// activation wrapper
typedef struct _nnom_activation_t nnom_activation_t;

typedef struct _nnom_buf
{
	nnom_mem_block_t *mem;
	size_t size;
	uint8_t type;
} nnom_buf_t;

// a memory block to store pre-assign memories during compiling. then assigned to each tensor after.
struct _nnom_mem_block_t
{
	void *blk;		// data block location
	size_t size;	// the maximum size for this block
	uint8_t owners; // how many layers own this block
	uint8_t state;  // empty? filled? for static nn, currently only used in compiling
};

typedef struct _nnom_stat_t
{
	size_t macc; //num. of mac operation
	uint32_t time;
} nnom_layer_stat_t;

struct _nnom_layer_hook_t
{
	nnom_layer_io_t *io;	 // hooked io
	nnom_layer_hook_t *next; // next hook include secondary hooked layer
};

struct _nnom_layer_io_t
{
	nnom_layer_hook_t hook;		  // for example: (layer->out)--hook--(layer->in)
	nnom_layer_io_t *aux; 			// point to auxilary I/O (multiple I/O layer)
	nnom_tensor_t *tensor;		  // experimental 
	nnom_mem_block_t *mem;		  // memory blocks handles for compiling only. The memory are now pass by tensor. trying to remove it. 
	nnom_layer_t *owner;		  // which layer owns this io.
	uint8_t type;
};

// structured configuration base type
typedef struct _nnom_layer_config_t
{
	char* name;			// the name of the layer prequantiesd model (the model trained by user before converted to nnom)
} nnom_layer_config_t;

// layers base
struct _nnom_layer_t
{
	nnom_layer_t *shortcut; // shortcut points to the next layer, applied on compiling

	nnom_status_t (*run)(nnom_layer_t *layer);				// run method. required
	nnom_status_t (*build)(nnom_layer_t *layer);			// compute output buffer shape. can be left null, will call default_build()
	nnom_status_t (*free)(nnom_layer_t *layer);				// a callback to free private resources (comp buf not included) can be left null
	nnom_buf_t *comp;		   								// computational buf
	nnom_activation_t *actail; 								// I have an activation, I have a tail, wooo haaaa, act-tail!!!

	nnom_layer_config_t *config;			// point to the configuration of the layers. for machine api only. 
	nnom_layer_type_t type; // layer types
	nnom_layer_io_t *in;	// IO buff, last*layer, states
	nnom_layer_io_t *out;   // IO buff, next*layer, states
	nnom_layer_stat_t stat; // stats, timing, ops
};

// activation base, super class must be the first member of the sub class
struct _nnom_activation_t
{
	nnom_status_t (*run)(struct _nnom_activation_t *act);
	nnom_tensor_t *tensor;
	nnom_activation_type_t type;
};

// local static functions when libc is not available
#ifdef NNOM_USING_STATIC_MEMORY
    void nnom_set_static_buf(void* buf, size_t size);
    void *nnom_malloc(size_t size);
    void nnom_free(void* p);
#endif //NNOM_USING_STATIC_BUF

typedef struct _nnom_model nnom_model_t;

#include "nnom_tensor.h"
#include "nnom_layers.h"
#include "nnom_utils.h"

// models, I dont want to make model class as a child of layer class yet
struct _nnom_model
{
	nnom_layer_t *head;
	nnom_layer_t *tail;

	// model constructor
	nnom_status_t (*add)(struct _nnom_model *m, nnom_layer_t *layer);					// has too pass a raw value
	nnom_layer_t *(*hook)(nnom_layer_t *curr, nnom_layer_t *last);						// create hook between 2 layers' primary IO.
	nnom_layer_t *(*merge)(nnom_layer_t *method, nnom_layer_t *in1, nnom_layer_t *in2); // an older interface of merge 2 inputs.
	nnom_layer_t *(*mergex)(nnom_layer_t *method, int num, ...);						// merge a few layers using mutiple input method (concate, add, ...)
	nnom_layer_t *(*active)(nnom_activation_t *act, nnom_layer_t *target_layer);		// add the activation to the existing layer's tail

	// callback
	nnom_status_t (*layer_callback)(nnom_model_t *m, nnom_layer_t *layer);				// layer callback will be called after each layer(after actail). 

	// block memory for layers
	nnom_mem_block_t blocks[NNOM_BLOCK_NUM];

	size_t total_ops;

	bool is_inited; 	//	is this structure initialized
	bool is_allocated;  //	is this structure allocated by nnom (not by user)
};

#define NNOM_NULL_CHECK(p)                 \
	if ((p) == NULL)                       \
	{                                 	   \
		NNOM_LOG("Error: NULL object.\n"); \
		return NN_ARGUMENT_ERROR;          \
	}


// utils
size_t nnom_alignto(size_t value, uint32_t alignment);
size_t nnom_io_length(nnom_layer_io_t *io);
size_t nnom_hook_length(nnom_layer_hook_t *hook);

// memory (malloc + memeset 0)
void *nnom_mem(size_t size);
	
// get how much memory has been taken
size_t nnom_mem_stat(void);

// Model APIs
// create or init a model
nnom_model_t *new_model(nnom_model_t *m);
// compile as sequencial model
nnom_status_t sequencial_compile(nnom_model_t *m);
// compile as functional model
nnom_status_t model_compile(nnom_model_t *m, nnom_layer_t *input, nnom_layer_t *output);
// run a prediction
nnom_status_t model_run(nnom_model_t *m);
// delete model. 
void model_delete(nnom_model_t *m);
// check version
nnom_status_t check_model_version(unsigned long model_version);

// callback, called after each layer has finished the calculation. 
// this callback must return NN_SUCCESS for continually run the model. otherwise, model will be returned with the ERROR code. 
// this function return NN_LENGTH_ERROR if the callback is already set to other. 
nnom_status_t model_set_callback(nnom_model_t *m, nnom_status_t (*layer_callback)(nnom_model_t *m, nnom_layer_t *layer));
// delete callback. 
void model_delete_callback(nnom_model_t *m);

#ifdef __cplusplus
}
#endif

#endif /* __NNOM_H__ */
