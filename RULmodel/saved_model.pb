۹4
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.4.12v2.4.1-0-g85c8b2a817f8??1
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_1/lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P**
shared_namelstm_1/lstm_cell_1/kernel
?
-lstm_1/lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_1/kernel*
_output_shapes

:P*
dtype0
?
#lstm_1/lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*4
shared_name%#lstm_1/lstm_cell_1/recurrent_kernel
?
7lstm_1/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_1/lstm_cell_1/recurrent_kernel*
_output_shapes

:P*
dtype0
?
lstm_1/lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*(
shared_namelstm_1/lstm_cell_1/bias

+lstm_1/lstm_cell_1/bias/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_1/bias*
_output_shapes
:P*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/m
?
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/m
?
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
 Adam/lstm_1/lstm_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*1
shared_name" Adam/lstm_1/lstm_cell_1/kernel/m
?
4Adam/lstm_1/lstm_cell_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_1/lstm_cell_1/kernel/m*
_output_shapes

:P*
dtype0
?
*Adam/lstm_1/lstm_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*;
shared_name,*Adam/lstm_1/lstm_cell_1/recurrent_kernel/m
?
>Adam/lstm_1/lstm_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_1/lstm_cell_1/recurrent_kernel/m*
_output_shapes

:P*
dtype0
?
Adam/lstm_1/lstm_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*/
shared_name Adam/lstm_1/lstm_cell_1/bias/m
?
2Adam/lstm_1/lstm_cell_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_1/lstm_cell_1/bias/m*
_output_shapes
:P*
dtype0
?
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/v
?
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/v
?
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
 Adam/lstm_1/lstm_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*1
shared_name" Adam/lstm_1/lstm_cell_1/kernel/v
?
4Adam/lstm_1/lstm_cell_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_1/lstm_cell_1/kernel/v*
_output_shapes

:P*
dtype0
?
*Adam/lstm_1/lstm_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*;
shared_name,*Adam/lstm_1/lstm_cell_1/recurrent_kernel/v
?
>Adam/lstm_1/lstm_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_1/lstm_cell_1/recurrent_kernel/v*
_output_shapes

:P*
dtype0
?
Adam/lstm_1/lstm_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*/
shared_name Adam/lstm_1/lstm_cell_1/bias/v
?
2Adam/lstm_1/lstm_cell_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_1/lstm_cell_1/bias/v*
_output_shapes
:P*
dtype0

NoOpNoOp
?1
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?1
value?1B?1 B?1
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
R
trainable_variables
	variables
regularization_losses
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
R
%trainable_variables
&	variables
'regularization_losses
(	keras_api
?
)iter

*beta_1

+beta_2
	,decay
-learning_ratem^m_m` ma.mb/mc0mdvevfvg vh.vi/vj0vk
1
0
1
.2
/3
04
5
 6
?
0
1
2
3
.4
/5
06
7
 8
 
?
trainable_variables
1non_trainable_variables
	variables
2metrics
	regularization_losses
3layer_metrics

4layers
5layer_regularization_losses
 
 
 
 
?
trainable_variables
6non_trainable_variables
	variables
7metrics
regularization_losses
8layer_metrics

9layers
:layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
2
3
 
?
trainable_variables
;non_trainable_variables
	variables
<metrics
regularization_losses
=layer_metrics

>layers
?layer_regularization_losses
~

.kernel
/recurrent_kernel
0bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
 

.0
/1
02

.0
/1
02
 
?

Dstates
trainable_variables
Enon_trainable_variables
	variables
Fmetrics
regularization_losses
Glayer_metrics

Hlayers
Ilayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
!trainable_variables
Jnon_trainable_variables
"	variables
Kmetrics
#regularization_losses
Llayer_metrics

Mlayers
Nlayer_regularization_losses
 
 
 
?
%trainable_variables
Onon_trainable_variables
&	variables
Pmetrics
'regularization_losses
Qlayer_metrics

Rlayers
Slayer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_1/lstm_cell_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_1/lstm_cell_1/recurrent_kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_1/lstm_cell_1/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE

0
1

T0
 
#
0
1
2
3
4
 
 
 
 
 
 

0
1
 
 
 
 

.0
/1
02

.0
/1
02
 
?
@trainable_variables
Unon_trainable_variables
A	variables
Vmetrics
Bregularization_losses
Wlayer_metrics

Xlayers
Ylayer_regularization_losses
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
4
	Ztotal
	[count
\	variables
]	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1

\	variables
??
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_1/lstm_cell_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_1/lstm_cell_1/recurrent_kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_1/lstm_cell_1/bias/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_1/lstm_cell_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_1/lstm_cell_1/recurrent_kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_1/lstm_cell_1/bias/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_masking_1_inputPlaceholder*4
_output_shapes"
 :??????????????????*
dtype0*)
shape :??????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_masking_1_input%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betalstm_1/lstm_cell_1/kernellstm_1/lstm_cell_1/bias#lstm_1/lstm_cell_1/recurrent_kerneldense_1/kerneldense_1/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_14161860
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_1/lstm_cell_1/kernel/Read/ReadVariableOp7lstm_1/lstm_cell_1/recurrent_kernel/Read/ReadVariableOp+lstm_1/lstm_cell_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp4Adam/lstm_1/lstm_cell_1/kernel/m/Read/ReadVariableOp>Adam/lstm_1/lstm_cell_1/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_1/lstm_cell_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp4Adam/lstm_1/lstm_cell_1/kernel/v/Read/ReadVariableOp>Adam/lstm_1/lstm_cell_1/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_1/lstm_cell_1/bias/v/Read/ReadVariableOpConst"/device:CPU:0*-
dtypes#
!2	
?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOpAssignVariableOpbatch_normalization_1/gamma
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_1AssignVariableOpbatch_normalization_1/beta
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_2AssignVariableOp!batch_normalization_1/moving_mean
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_3AssignVariableOp%batch_normalization_1/moving_variance
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_4AssignVariableOpdense_1/kernel
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_5AssignVariableOpdense_1/bias
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0	*
_output_shapes
:
Y
AssignVariableOp_6AssignVariableOp	Adam/iter
Identity_7"/device:CPU:0*
dtype0	
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_7AssignVariableOpAdam/beta_1
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_8AssignVariableOpAdam/beta_2
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_9AssignVariableOp
Adam/decayIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOp_10AssignVariableOpAdam/learning_rateIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_11AssignVariableOplstm_1/lstm_cell_1/kernelIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_12AssignVariableOp#lstm_1/lstm_cell_1/recurrent_kernelIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_13AssignVariableOplstm_1/lstm_cell_1/biasIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_14AssignVariableOptotalIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_15AssignVariableOpcountIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_16AssignVariableOp"Adam/batch_normalization_1/gamma/mIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
s
AssignVariableOp_17AssignVariableOp!Adam/batch_normalization_1/beta/mIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_18AssignVariableOpAdam/dense_1/kernel/mIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_19AssignVariableOpAdam/dense_1/bias/mIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_20AssignVariableOp Adam/lstm_1/lstm_cell_1/kernel/mIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
|
AssignVariableOp_21AssignVariableOp*Adam/lstm_1/lstm_cell_1/recurrent_kernel/mIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
p
AssignVariableOp_22AssignVariableOpAdam/lstm_1/lstm_cell_1/bias/mIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_23AssignVariableOp"Adam/batch_normalization_1/gamma/vIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
s
AssignVariableOp_24AssignVariableOp!Adam/batch_normalization_1/beta/vIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_25AssignVariableOpAdam/dense_1/kernel/vIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_26AssignVariableOpAdam/dense_1/bias/vIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_27AssignVariableOp Adam/lstm_1/lstm_cell_1/kernel/vIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
|
AssignVariableOp_28AssignVariableOp*Adam/lstm_1/lstm_cell_1/recurrent_kernel/vIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
p
AssignVariableOp_29AssignVariableOpAdam/lstm_1/lstm_cell_1/bias/vIdentity_30"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
?
Identity_31Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ??/
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_14164300

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_14164335

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_1S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xg
mulMulmul/x:output:0strided_slice:output:0*
T0*#
_output_shapes
:?????????2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/xo
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*#
_output_shapes
:?????????2
mul_1H
ExpExpmul:z:0*
T0*#
_output_shapes
:?????????2
ExpW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?H2	
mul_2/x^
mul_2Mulmul_2/x:output:0Exp:y:0*
T0*#
_output_shapes
:?????????2
mul_2S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *9?@2
sub/yZ
subSub	mul_1:z:0sub/y:output:0*
T0*#
_output_shapes
:?????????2
subT
SigmoidSigmoidsub:z:0*
T0*#
_output_shapes
:?????????2	
SigmoidW
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2	
mul_3/xb
mul_3Mulmul_3/x:output:0Sigmoid:y:0*
T0*#
_output_shapes
:?????????2
mul_3|
stackPack	mul_2:z:0	mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*
axis?????????2
stackb
IdentityIdentitystack:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
)__inference_lstm_1_layer_call_fn_14162845
inputs_0-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
lstm_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout/Const?
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Mul?
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shape?
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???22
0lstm_cell_1/dropout/random_uniform/RandomUniform?
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2$
"lstm_cell_1/dropout/GreaterEqual/y?
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_cell_1/dropout/GreaterEqual?
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Cast?
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_1/Const?
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Mul?
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shape?
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??:24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform?
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_1/GreaterEqual/y?
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_1/GreaterEqual?
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Cast?
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_2/Const?
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Mul?
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shape?
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2됃24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform?
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_2/GreaterEqual/y?
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_2/GreaterEqual?
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Cast?
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_3/Const?
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Mul?
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shape?
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ř24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform?
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_3/GreaterEqual/y?
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_3/GreaterEqual?
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Cast?
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Mul_1h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mulMulzeros:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulzeros:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulzeros:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulzeros:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_3?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add|
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_4?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_2u
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_5?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_2y
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_6?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_14162684*
condR
while_cond_14162683*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_14162137
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_14162137___redundant_placeholder06
2while_while_cond_14162137___redundant_placeholder16
2while_while_cond_14162137___redundant_placeholder26
2while_while_cond_14162137___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?
while_body_14162957
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/ones_liket
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mulMulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_3?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_4?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_6?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_1/mul_6:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
lstm_1_while_cond_14160461*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4,
(lstm_1_while_less_lstm_1_strided_slice_1D
@lstm_1_while_lstm_1_while_cond_14160461___redundant_placeholder0D
@lstm_1_while_lstm_1_while_cond_14160461___redundant_placeholder1D
@lstm_1_while_lstm_1_while_cond_14160461___redundant_placeholder2D
@lstm_1_while_lstm_1_while_cond_14160461___redundant_placeholder3D
@lstm_1_while_lstm_1_while_cond_14160461___redundant_placeholder4
lstm_1_while_identity
?
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: 2
lstm_1/while/Lessr
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_1/while/Identity"7
lstm_1_while_identitylstm_1/while/Identity:output:0*j
_input_shapesY
W: : : : :?????????:?????????:?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
??
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_14161110
masking_1_input;
7batch_normalization_1_batchnorm_readvariableop_resource?
;batch_normalization_1_batchnorm_mul_readvariableop_resource=
9batch_normalization_1_batchnorm_readvariableop_1_resource=
9batch_normalization_1_batchnorm_readvariableop_2_resource4
0lstm_1_lstm_cell_1_split_readvariableop_resource6
2lstm_1_lstm_cell_1_split_1_readvariableop_resource.
*lstm_1_lstm_cell_1_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??.batch_normalization_1/batchnorm/ReadVariableOp?0batch_normalization_1/batchnorm/ReadVariableOp_1?0batch_normalization_1/batchnorm/ReadVariableOp_2?2batch_normalization_1/batchnorm/mul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?!lstm_1/lstm_cell_1/ReadVariableOp?#lstm_1/lstm_cell_1/ReadVariableOp_1?#lstm_1/lstm_cell_1/ReadVariableOp_2?#lstm_1/lstm_cell_1/ReadVariableOp_3?'lstm_1/lstm_cell_1/split/ReadVariableOp?)lstm_1/lstm_cell_1/split_1/ReadVariableOp?lstm_1/whileq
masking_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
masking_1/NotEqual/y?
masking_1/NotEqualNotEqualmasking_1_inputmasking_1/NotEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
masking_1/NotEqual?
masking_1/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
masking_1/Any/reduction_indices?
masking_1/AnyAnymasking_1/NotEqual:z:0(masking_1/Any/reduction_indices:output:0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
masking_1/Any?
masking_1/CastCastmasking_1/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
masking_1/Cast?
masking_1/mulMulmasking_1_inputmasking_1/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
masking_1/mul?
masking_1/SqueezeSqueezemasking_1/Any:output:0*
T0
*0
_output_shapes
:??????????????????*
squeeze_dims

?????????2
masking_1/Squeeze?
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/Rsqrt?
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Mulmasking_1/mul:z:0'batch_normalization_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2'
%batch_normalization_1/batchnorm/mul_1?
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1?
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/mul_2?
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2?
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2'
%batch_normalization_1/batchnorm/add_1u
lstm_1/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
lstm_1/Shape?
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice/stack?
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_1?
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_2?
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slicej
lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros/mul/y?
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/mulm
lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros/Less/y?
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/Lessp
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros/packed/1?
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros/packedm
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros/Const?
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/zerosn
lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros_1/mul/y?
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/mulq
lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros_1/Less/y?
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/Lesst
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros_1/packed/1?
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros_1/packedq
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros_1/Const?
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/zeros_1?
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose/perm?
lstm_1/transpose	Transpose)batch_normalization_1/batchnorm/add_1:z:0lstm_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_1/transposed
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:2
lstm_1/Shape_1?
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_1/stack?
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_1?
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_2?
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slice_1y
lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_1/ExpandDims/dim?
lstm_1/ExpandDims
ExpandDimsmasking_1/Squeeze:output:0lstm_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????2
lstm_1/ExpandDims?
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_1/perm?
lstm_1/transpose_1	Transposelstm_1/ExpandDims:output:0 lstm_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????2
lstm_1/transpose_1?
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_1/TensorArrayV2/element_shape?
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2?
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_1/TensorArrayUnstack/TensorListFromTensor?
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_2/stack?
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_1?
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_2?
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_1/strided_slice_2?
"lstm_1/lstm_cell_1/ones_like/ShapeShapelstm_1/zeros:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/ones_like/Shape?
"lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"lstm_1/lstm_cell_1/ones_like/Const?
lstm_1/lstm_cell_1/ones_likeFill+lstm_1/lstm_cell_1/ones_like/Shape:output:0+lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/ones_likev
lstm_1/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const?
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_1/lstm_cell_1/split/split_dim?
'lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02)
'lstm_1/lstm_cell_1/split/ReadVariableOp?
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_1/lstm_cell_1/split?
lstm_1/lstm_cell_1/MatMulMatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul?
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_1?
lstm_1/lstm_cell_1/MatMul_2MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_2?
lstm_1/lstm_cell_1/MatMul_3MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_3z
lstm_1/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const_1?
$lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_1/lstm_cell_1/split_1/split_dim?
)lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02+
)lstm_1/lstm_cell_1/split_1/ReadVariableOp?
lstm_1/lstm_cell_1/split_1Split-lstm_1/lstm_cell_1/split_1/split_dim:output:01lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_1/lstm_cell_1/split_1?
lstm_1/lstm_cell_1/BiasAddBiasAdd#lstm_1/lstm_cell_1/MatMul:product:0#lstm_1/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd?
lstm_1/lstm_cell_1/BiasAdd_1BiasAdd%lstm_1/lstm_cell_1/MatMul_1:product:0#lstm_1/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_1?
lstm_1/lstm_cell_1/BiasAdd_2BiasAdd%lstm_1/lstm_cell_1/MatMul_2:product:0#lstm_1/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_2?
lstm_1/lstm_cell_1/BiasAdd_3BiasAdd%lstm_1/lstm_cell_1/MatMul_3:product:0#lstm_1/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_3?
lstm_1/lstm_cell_1/mulMullstm_1/zeros:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul?
lstm_1/lstm_cell_1/mul_1Mullstm_1/zeros:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_1?
lstm_1/lstm_cell_1/mul_2Mullstm_1/zeros:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_2?
lstm_1/lstm_cell_1/mul_3Mullstm_1/zeros:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_3?
!lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02#
!lstm_1/lstm_cell_1/ReadVariableOp?
&lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_1/lstm_cell_1/strided_slice/stack?
(lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice/stack_1?
(lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell_1/strided_slice/stack_2?
 lstm_1/lstm_cell_1/strided_sliceStridedSlice)lstm_1/lstm_cell_1/ReadVariableOp:value:0/lstm_1/lstm_cell_1/strided_slice/stack:output:01lstm_1/lstm_cell_1/strided_slice/stack_1:output:01lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2"
 lstm_1/lstm_cell_1/strided_slice?
lstm_1/lstm_cell_1/MatMul_4MatMullstm_1/lstm_cell_1/mul:z:0)lstm_1/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_4?
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/BiasAdd:output:0%lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add?
lstm_1/lstm_cell_1/SigmoidSigmoidlstm_1/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid?
#lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_1?
(lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice_1/stack?
*lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_1?
*lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_2?
"lstm_1/lstm_cell_1/strided_slice_1StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_1:value:01lstm_1/lstm_cell_1/strided_slice_1/stack:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_1?
lstm_1/lstm_cell_1/MatMul_5MatMullstm_1/lstm_cell_1/mul_1:z:0+lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_5?
lstm_1/lstm_cell_1/add_1AddV2%lstm_1/lstm_cell_1/BiasAdd_1:output:0%lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_1?
lstm_1/lstm_cell_1/Sigmoid_1Sigmoidlstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid_1?
lstm_1/lstm_cell_1/mul_4Mul lstm_1/lstm_cell_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_4?
#lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_2?
(lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2*
(lstm_1/lstm_cell_1/strided_slice_2/stack?
*lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_1?
*lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_2?
"lstm_1/lstm_cell_1/strided_slice_2StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_2:value:01lstm_1/lstm_cell_1/strided_slice_2/stack:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_2?
lstm_1/lstm_cell_1/MatMul_6MatMullstm_1/lstm_cell_1/mul_2:z:0+lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_6?
lstm_1/lstm_cell_1/add_2AddV2%lstm_1/lstm_cell_1/BiasAdd_2:output:0%lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_2?
lstm_1/lstm_cell_1/TanhTanhlstm_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Tanh?
lstm_1/lstm_cell_1/mul_5Mullstm_1/lstm_cell_1/Sigmoid:y:0lstm_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_5?
lstm_1/lstm_cell_1/add_3AddV2lstm_1/lstm_cell_1/mul_4:z:0lstm_1/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_3?
#lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_3?
(lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2*
(lstm_1/lstm_cell_1/strided_slice_3/stack?
*lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_1?
*lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_2?
"lstm_1/lstm_cell_1/strided_slice_3StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_3:value:01lstm_1/lstm_cell_1/strided_slice_3/stack:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_3?
lstm_1/lstm_cell_1/MatMul_7MatMullstm_1/lstm_cell_1/mul_3:z:0+lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_7?
lstm_1/lstm_cell_1/add_4AddV2%lstm_1/lstm_cell_1/BiasAdd_3:output:0%lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_4?
lstm_1/lstm_cell_1/Sigmoid_2Sigmoidlstm_1/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid_2?
lstm_1/lstm_cell_1/Tanh_1Tanhlstm_1/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Tanh_1?
lstm_1/lstm_cell_1/mul_6Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_6?
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$lstm_1/TensorArrayV2_1/element_shape?
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2_1\
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/time?
$lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$lstm_1/TensorArrayV2_2/element_shape?
lstm_1/TensorArrayV2_2TensorListReserve-lstm_1/TensorArrayV2_2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
lstm_1/TensorArrayV2_2?
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
0lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm_1/transpose_1:y:0Glstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type022
0lstm_1/TensorArrayUnstack_1/TensorListFromTensor?
lstm_1/zeros_like	ZerosLikelstm_1/lstm_cell_1/mul_6:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/zeros_like?
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_1/while/maximum_iterationsx
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/while/loop_counter?
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros_like:y:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:00lstm_1_lstm_cell_1_split_readvariableop_resource2lstm_1_lstm_cell_1_split_1_readvariableop_resource*lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *%
_read_only_resource_inputs

*&
bodyR
lstm_1_while_body_14160936*&
condR
lstm_1_while_cond_14160935*`
output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *
parallel_iterations 2
lstm_1/while?
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02+
)lstm_1/TensorArrayV2Stack/TensorListStack?
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_1/strided_slice_3/stack?
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_1/strided_slice_3/stack_1?
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_3/stack_2?
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_1/strided_slice_3?
lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_2/perm?
lstm_1/transpose_2	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_1/transpose_2t
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/runtime?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMullstm_1/strided_slice_3:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lambda_1/strided_slice/stack?
lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_1/strided_slice/stack_1?
lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_1/strided_slice/stack_2?
lambda_1/strided_sliceStridedSlicedense_1/BiasAdd:output:0%lambda_1/strided_slice/stack:output:0'lambda_1/strided_slice/stack_1:output:0'lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
lambda_1/strided_slice?
lambda_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_1/strided_slice_1/stack?
 lambda_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 lambda_1/strided_slice_1/stack_1?
 lambda_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 lambda_1/strided_slice_1/stack_2?
lambda_1/strided_slice_1StridedSlicedense_1/BiasAdd:output:0'lambda_1/strided_slice_1/stack:output:0)lambda_1/strided_slice_1/stack_1:output:0)lambda_1/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
lambda_1/strided_slice_1e
lambda_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda_1/mul/x?
lambda_1/mulMullambda_1/mul/x:output:0lambda_1/strided_slice:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/muli
lambda_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda_1/mul_1/x?
lambda_1/mul_1Mullambda_1/mul_1/x:output:0!lambda_1/strided_slice_1:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_1c
lambda_1/ExpExplambda_1/mul:z:0*
T0*#
_output_shapes
:?????????2
lambda_1/Expi
lambda_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?H2
lambda_1/mul_2/x?
lambda_1/mul_2Mullambda_1/mul_2/x:output:0lambda_1/Exp:y:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_2e
lambda_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *9?@2
lambda_1/sub/y~
lambda_1/subSublambda_1/mul_1:z:0lambda_1/sub/y:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/subo
lambda_1/SigmoidSigmoidlambda_1/sub:z:0*
T0*#
_output_shapes
:?????????2
lambda_1/Sigmoidi
lambda_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2
lambda_1/mul_3/x?
lambda_1/mul_3Mullambda_1/mul_3/x:output:0lambda_1/Sigmoid:y:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_3?
lambda_1/stackPacklambda_1/mul_2:z:0lambda_1/mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*
axis?????????2
lambda_1/stack?
IdentityIdentitylambda_1/stack:output:0/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:??????????????????:::::::::2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2F
!lstm_1/lstm_cell_1/ReadVariableOp!lstm_1/lstm_cell_1/ReadVariableOp2J
#lstm_1/lstm_cell_1/ReadVariableOp_1#lstm_1/lstm_cell_1/ReadVariableOp_12J
#lstm_1/lstm_cell_1/ReadVariableOp_2#lstm_1/lstm_cell_1/ReadVariableOp_22J
#lstm_1/lstm_cell_1/ReadVariableOp_3#lstm_1/lstm_cell_1/ReadVariableOp_32R
'lstm_1/lstm_cell_1/split/ReadVariableOp'lstm_1/lstm_cell_1/split/ReadVariableOp2V
)lstm_1/lstm_cell_1/split_1/ReadVariableOp)lstm_1/lstm_cell_1/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:e a
4
_output_shapes"
 :??????????????????
)
_user_specified_namemasking_1_input
??
?
)__inference_lstm_1_layer_call_fn_14164290

inputs
mask
-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????2
transpose_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
lstm_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/ones_likeh
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mulMulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_3?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add|
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_4?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_2u
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_5?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_2y
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_6?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time?
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2_2/element_shape?
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2?
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensorn

zeros_like	ZerosLikelstm_cell_1/mul_6:z:0*
T0*'
_output_shapes
:?????????2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *%
_read_only_resource_inputs

*
bodyR
while_body_14164143*
condR
while_cond_14164142*`
output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????????????:??????????????????:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
G
+__inference_lambda_1_layer_call_fn_14164385

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_1S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xg
mulMulmul/x:output:0strided_slice:output:0*
T0*#
_output_shapes
:?????????2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/xo
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*#
_output_shapes
:?????????2
mul_1H
ExpExpmul:z:0*
T0*#
_output_shapes
:?????????2
ExpW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?H2	
mul_2/x^
mul_2Mulmul_2/x:output:0Exp:y:0*
T0*#
_output_shapes
:?????????2
mul_2S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *9?@2
sub/yZ
subSub	mul_1:z:0sub/y:output:0*
T0*#
_output_shapes
:?????????2
subT
SigmoidSigmoidsub:z:0*
T0*#
_output_shapes
:?????????2	
SigmoidW
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2	
mul_3/xb
mul_3Mulmul_3/x:output:0Sigmoid:y:0*
T0*#
_output_shapes
:?????????2
mul_3|
stackPack	mul_2:z:0	mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*
axis?????????2
stackb
IdentityIdentitystack:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
c
G__inference_masking_1_layer_call_and_return_conditional_losses_14161871

inputs
identity]

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

NotEqual/y|
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Any/reduction_indices?
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Anyp
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
Castb
mulMulinputsCast:y:0*
T0*4
_output_shapes"
 :??????????????????2
mul?
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:??????????????????*
squeeze_dims

?????????2	
Squeezeh
IdentityIdentitymul:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
D__inference_lstm_1_layer_call_and_return_conditional_losses_14162540
inputs_0-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
lstm_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/ones_likeh
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mulMulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_3?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add|
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_4?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_2u
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_5?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_2y
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_6?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_14162411*
condR
while_cond_14162410*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
??
?

while_body_14163842
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/ones_like?
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2!
while/lstm_cell_1/dropout/Const?
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/dropout/Mul?
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_1/dropout/Shape?
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell_1/dropout/random_uniform/RandomUniform?
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2*
(while/lstm_cell_1/dropout/GreaterEqual/y?
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2(
&while/lstm_cell_1/dropout/GreaterEqual?
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2 
while/lstm_cell_1/dropout/Cast?
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout/Mul_1?
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_1/Const?
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_1/Mul?
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_1/Shape?
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_1/GreaterEqual/y?
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_1/GreaterEqual?
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_1/Cast?
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_1/Mul_1?
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_2/Const?
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_2/Mul?
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_2/Shape?
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_2/GreaterEqual/y?
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_2/GreaterEqual?
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_2/Cast?
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_2/Mul_1?
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_3/Const?
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_3/Mul?
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_3/Shape?
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_3/GreaterEqual/y?
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_3/GreaterEqual?
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_3/Cast?
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_3/Mul_1t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mulMulwhile_placeholder_3#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mulwhile_placeholder_3%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mulwhile_placeholder_3%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mulwhile_placeholder_3%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_3?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_4*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_4?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_6}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2

while/Tile?
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/SelectV2?
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_1?
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_2?
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/SelectV2_1?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_1/add_3:z:0while_placeholder_4*
T0*'
_output_shapes
:?????????2
while/SelectV2_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/SelectV2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/SelectV2_1:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5?
while/Identity_6Identitywhile/SelectV2_2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_6")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :?????????:?????????:?????????: : : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
?
while_cond_14162956
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_14162956___redundant_placeholder06
2while_while_cond_14162956___redundant_placeholder16
2while_while_cond_14162956___redundant_placeholder26
2while_while_cond_14162956___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?
)__inference_lstm_1_layer_call_fn_14163086
inputs_0-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
lstm_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/ones_likeh
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mulMulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_3?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add|
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_4?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_2u
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_5?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_2y
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_6?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_14162957*
condR
while_cond_14162956*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
˹
?

#__inference__wrapped_model_14157347
masking_1_inputH
Dsequential_1_batch_normalization_1_batchnorm_readvariableop_resourceL
Hsequential_1_batch_normalization_1_batchnorm_mul_readvariableop_resourceJ
Fsequential_1_batch_normalization_1_batchnorm_readvariableop_1_resourceJ
Fsequential_1_batch_normalization_1_batchnorm_readvariableop_2_resourceA
=sequential_1_lstm_1_lstm_cell_1_split_readvariableop_resourceC
?sequential_1_lstm_1_lstm_cell_1_split_1_readvariableop_resource;
7sequential_1_lstm_1_lstm_cell_1_readvariableop_resource7
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity??;sequential_1/batch_normalization_1/batchnorm/ReadVariableOp?=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1?=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2??sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?.sequential_1/lstm_1/lstm_cell_1/ReadVariableOp?0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_1?0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_2?0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_3?4sequential_1/lstm_1/lstm_cell_1/split/ReadVariableOp?6sequential_1/lstm_1/lstm_cell_1/split_1/ReadVariableOp?sequential_1/lstm_1/while?
!sequential_1/masking_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!sequential_1/masking_1/NotEqual/y?
sequential_1/masking_1/NotEqualNotEqualmasking_1_input*sequential_1/masking_1/NotEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2!
sequential_1/masking_1/NotEqual?
,sequential_1/masking_1/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_1/masking_1/Any/reduction_indices?
sequential_1/masking_1/AnyAny#sequential_1/masking_1/NotEqual:z:05sequential_1/masking_1/Any/reduction_indices:output:0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
sequential_1/masking_1/Any?
sequential_1/masking_1/CastCast#sequential_1/masking_1/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
sequential_1/masking_1/Cast?
sequential_1/masking_1/mulMulmasking_1_inputsequential_1/masking_1/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
sequential_1/masking_1/mul?
sequential_1/masking_1/SqueezeSqueeze#sequential_1/masking_1/Any:output:0*
T0
*0
_output_shapes
:??????????????????*
squeeze_dims

?????????2 
sequential_1/masking_1/Squeeze?
;sequential_1/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpDsequential_1_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_1/batch_normalization_1/batchnorm/ReadVariableOp?
2sequential_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_1/batch_normalization_1/batchnorm/add/y?
0sequential_1/batch_normalization_1/batchnorm/addAddV2Csequential_1/batch_normalization_1/batchnorm/ReadVariableOp:value:0;sequential_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_1/batch_normalization_1/batchnorm/add?
2sequential_1/batch_normalization_1/batchnorm/RsqrtRsqrt4sequential_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_1/batch_normalization_1/batchnorm/Rsqrt?
?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_1_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp?
0sequential_1/batch_normalization_1/batchnorm/mulMul6sequential_1/batch_normalization_1/batchnorm/Rsqrt:y:0Gsequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_1/batch_normalization_1/batchnorm/mul?
2sequential_1/batch_normalization_1/batchnorm/mul_1Mulsequential_1/masking_1/mul:z:04sequential_1/batch_normalization_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????24
2sequential_1/batch_normalization_1/batchnorm/mul_1?
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_1_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1?
2sequential_1/batch_normalization_1/batchnorm/mul_2MulEsequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1:value:04sequential_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_1/batch_normalization_1/batchnorm/mul_2?
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_1_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02?
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2?
0sequential_1/batch_normalization_1/batchnorm/subSubEsequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2:value:06sequential_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_1/batch_normalization_1/batchnorm/sub?
2sequential_1/batch_normalization_1/batchnorm/add_1AddV26sequential_1/batch_normalization_1/batchnorm/mul_1:z:04sequential_1/batch_normalization_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????24
2sequential_1/batch_normalization_1/batchnorm/add_1?
sequential_1/lstm_1/ShapeShape6sequential_1/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
sequential_1/lstm_1/Shape?
'sequential_1/lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_1/lstm_1/strided_slice/stack?
)sequential_1/lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_1/strided_slice/stack_1?
)sequential_1/lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_1/strided_slice/stack_2?
!sequential_1/lstm_1/strided_sliceStridedSlice"sequential_1/lstm_1/Shape:output:00sequential_1/lstm_1/strided_slice/stack:output:02sequential_1/lstm_1/strided_slice/stack_1:output:02sequential_1/lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_1/lstm_1/strided_slice?
sequential_1/lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_1/lstm_1/zeros/mul/y?
sequential_1/lstm_1/zeros/mulMul*sequential_1/lstm_1/strided_slice:output:0(sequential_1/lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_1/zeros/mul?
 sequential_1/lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_1/lstm_1/zeros/Less/y?
sequential_1/lstm_1/zeros/LessLess!sequential_1/lstm_1/zeros/mul:z:0)sequential_1/lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_1/lstm_1/zeros/Less?
"sequential_1/lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_1/lstm_1/zeros/packed/1?
 sequential_1/lstm_1/zeros/packedPack*sequential_1/lstm_1/strided_slice:output:0+sequential_1/lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_1/lstm_1/zeros/packed?
sequential_1/lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/lstm_1/zeros/Const?
sequential_1/lstm_1/zerosFill)sequential_1/lstm_1/zeros/packed:output:0(sequential_1/lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential_1/lstm_1/zeros?
!sequential_1/lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_1/lstm_1/zeros_1/mul/y?
sequential_1/lstm_1/zeros_1/mulMul*sequential_1/lstm_1/strided_slice:output:0*sequential_1/lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_1/zeros_1/mul?
"sequential_1/lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_1/lstm_1/zeros_1/Less/y?
 sequential_1/lstm_1/zeros_1/LessLess#sequential_1/lstm_1/zeros_1/mul:z:0+sequential_1/lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_1/lstm_1/zeros_1/Less?
$sequential_1/lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_1/lstm_1/zeros_1/packed/1?
"sequential_1/lstm_1/zeros_1/packedPack*sequential_1/lstm_1/strided_slice:output:0-sequential_1/lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/lstm_1/zeros_1/packed?
!sequential_1/lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/lstm_1/zeros_1/Const?
sequential_1/lstm_1/zeros_1Fill+sequential_1/lstm_1/zeros_1/packed:output:0*sequential_1/lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential_1/lstm_1/zeros_1?
"sequential_1/lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_1/lstm_1/transpose/perm?
sequential_1/lstm_1/transpose	Transpose6sequential_1/batch_normalization_1/batchnorm/add_1:z:0+sequential_1/lstm_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
sequential_1/lstm_1/transpose?
sequential_1/lstm_1/Shape_1Shape!sequential_1/lstm_1/transpose:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_1/Shape_1?
)sequential_1/lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_1/strided_slice_1/stack?
+sequential_1/lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_1/strided_slice_1/stack_1?
+sequential_1/lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_1/strided_slice_1/stack_2?
#sequential_1/lstm_1/strided_slice_1StridedSlice$sequential_1/lstm_1/Shape_1:output:02sequential_1/lstm_1/strided_slice_1/stack:output:04sequential_1/lstm_1/strided_slice_1/stack_1:output:04sequential_1/lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_1/lstm_1/strided_slice_1?
"sequential_1/lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"sequential_1/lstm_1/ExpandDims/dim?
sequential_1/lstm_1/ExpandDims
ExpandDims'sequential_1/masking_1/Squeeze:output:0+sequential_1/lstm_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????2 
sequential_1/lstm_1/ExpandDims?
$sequential_1/lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_1/transpose_1/perm?
sequential_1/lstm_1/transpose_1	Transpose'sequential_1/lstm_1/ExpandDims:output:0-sequential_1/lstm_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????2!
sequential_1/lstm_1/transpose_1?
/sequential_1/lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_1/lstm_1/TensorArrayV2/element_shape?
!sequential_1/lstm_1/TensorArrayV2TensorListReserve8sequential_1/lstm_1/TensorArrayV2/element_shape:output:0,sequential_1/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_1/lstm_1/TensorArrayV2?
Isequential_1/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2K
Isequential_1/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_1/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_1/transpose:y:0Rsequential_1/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_1/lstm_1/TensorArrayUnstack/TensorListFromTensor?
)sequential_1/lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_1/strided_slice_2/stack?
+sequential_1/lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_1/strided_slice_2/stack_1?
+sequential_1/lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_1/strided_slice_2/stack_2?
#sequential_1/lstm_1/strided_slice_2StridedSlice!sequential_1/lstm_1/transpose:y:02sequential_1/lstm_1/strided_slice_2/stack:output:04sequential_1/lstm_1/strided_slice_2/stack_1:output:04sequential_1/lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2%
#sequential_1/lstm_1/strided_slice_2?
/sequential_1/lstm_1/lstm_cell_1/ones_like/ShapeShape"sequential_1/lstm_1/zeros:output:0*
T0*
_output_shapes
:21
/sequential_1/lstm_1/lstm_cell_1/ones_like/Shape?
/sequential_1/lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??21
/sequential_1/lstm_1/lstm_cell_1/ones_like/Const?
)sequential_1/lstm_1/lstm_cell_1/ones_likeFill8sequential_1/lstm_1/lstm_cell_1/ones_like/Shape:output:08sequential_1/lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2+
)sequential_1/lstm_1/lstm_cell_1/ones_like?
%sequential_1/lstm_1/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/lstm_1/lstm_cell_1/Const?
/sequential_1/lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/lstm_1/lstm_cell_1/split/split_dim?
4sequential_1/lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp=sequential_1_lstm_1_lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype026
4sequential_1/lstm_1/lstm_cell_1/split/ReadVariableOp?
%sequential_1/lstm_1/lstm_cell_1/splitSplit8sequential_1/lstm_1/lstm_cell_1/split/split_dim:output:0<sequential_1/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2'
%sequential_1/lstm_1/lstm_cell_1/split?
&sequential_1/lstm_1/lstm_cell_1/MatMulMatMul,sequential_1/lstm_1/strided_slice_2:output:0.sequential_1/lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2(
&sequential_1/lstm_1/lstm_cell_1/MatMul?
(sequential_1/lstm_1/lstm_cell_1/MatMul_1MatMul,sequential_1/lstm_1/strided_slice_2:output:0.sequential_1/lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2*
(sequential_1/lstm_1/lstm_cell_1/MatMul_1?
(sequential_1/lstm_1/lstm_cell_1/MatMul_2MatMul,sequential_1/lstm_1/strided_slice_2:output:0.sequential_1/lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2*
(sequential_1/lstm_1/lstm_cell_1/MatMul_2?
(sequential_1/lstm_1/lstm_cell_1/MatMul_3MatMul,sequential_1/lstm_1/strided_slice_2:output:0.sequential_1/lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2*
(sequential_1/lstm_1/lstm_cell_1/MatMul_3?
'sequential_1/lstm_1/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/lstm_1/lstm_cell_1/Const_1?
1sequential_1/lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_1/lstm_1/lstm_cell_1/split_1/split_dim?
6sequential_1/lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp?sequential_1_lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype028
6sequential_1/lstm_1/lstm_cell_1/split_1/ReadVariableOp?
'sequential_1/lstm_1/lstm_cell_1/split_1Split:sequential_1/lstm_1/lstm_cell_1/split_1/split_dim:output:0>sequential_1/lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2)
'sequential_1/lstm_1/lstm_cell_1/split_1?
'sequential_1/lstm_1/lstm_cell_1/BiasAddBiasAdd0sequential_1/lstm_1/lstm_cell_1/MatMul:product:00sequential_1/lstm_1/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2)
'sequential_1/lstm_1/lstm_cell_1/BiasAdd?
)sequential_1/lstm_1/lstm_cell_1/BiasAdd_1BiasAdd2sequential_1/lstm_1/lstm_cell_1/MatMul_1:product:00sequential_1/lstm_1/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2+
)sequential_1/lstm_1/lstm_cell_1/BiasAdd_1?
)sequential_1/lstm_1/lstm_cell_1/BiasAdd_2BiasAdd2sequential_1/lstm_1/lstm_cell_1/MatMul_2:product:00sequential_1/lstm_1/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2+
)sequential_1/lstm_1/lstm_cell_1/BiasAdd_2?
)sequential_1/lstm_1/lstm_cell_1/BiasAdd_3BiasAdd2sequential_1/lstm_1/lstm_cell_1/MatMul_3:product:00sequential_1/lstm_1/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2+
)sequential_1/lstm_1/lstm_cell_1/BiasAdd_3?
#sequential_1/lstm_1/lstm_cell_1/mulMul"sequential_1/lstm_1/zeros:output:02sequential_1/lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2%
#sequential_1/lstm_1/lstm_cell_1/mul?
%sequential_1/lstm_1/lstm_cell_1/mul_1Mul"sequential_1/lstm_1/zeros:output:02sequential_1/lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2'
%sequential_1/lstm_1/lstm_cell_1/mul_1?
%sequential_1/lstm_1/lstm_cell_1/mul_2Mul"sequential_1/lstm_1/zeros:output:02sequential_1/lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2'
%sequential_1/lstm_1/lstm_cell_1/mul_2?
%sequential_1/lstm_1/lstm_cell_1/mul_3Mul"sequential_1/lstm_1/zeros:output:02sequential_1/lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2'
%sequential_1/lstm_1/lstm_cell_1/mul_3?
.sequential_1/lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp7sequential_1_lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype020
.sequential_1/lstm_1/lstm_cell_1/ReadVariableOp?
3sequential_1/lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3sequential_1/lstm_1/lstm_cell_1/strided_slice/stack?
5sequential_1/lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5sequential_1/lstm_1/lstm_cell_1/strided_slice/stack_1?
5sequential_1/lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential_1/lstm_1/lstm_cell_1/strided_slice/stack_2?
-sequential_1/lstm_1/lstm_cell_1/strided_sliceStridedSlice6sequential_1/lstm_1/lstm_cell_1/ReadVariableOp:value:0<sequential_1/lstm_1/lstm_cell_1/strided_slice/stack:output:0>sequential_1/lstm_1/lstm_cell_1/strided_slice/stack_1:output:0>sequential_1/lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2/
-sequential_1/lstm_1/lstm_cell_1/strided_slice?
(sequential_1/lstm_1/lstm_cell_1/MatMul_4MatMul'sequential_1/lstm_1/lstm_cell_1/mul:z:06sequential_1/lstm_1/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2*
(sequential_1/lstm_1/lstm_cell_1/MatMul_4?
#sequential_1/lstm_1/lstm_cell_1/addAddV20sequential_1/lstm_1/lstm_cell_1/BiasAdd:output:02sequential_1/lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2%
#sequential_1/lstm_1/lstm_cell_1/add?
'sequential_1/lstm_1/lstm_cell_1/SigmoidSigmoid'sequential_1/lstm_1/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2)
'sequential_1/lstm_1/lstm_cell_1/Sigmoid?
0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp7sequential_1_lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype022
0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_1?
5sequential_1/lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5sequential_1/lstm_1/lstm_cell_1/strided_slice_1/stack?
7sequential_1/lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   29
7sequential_1/lstm_1/lstm_cell_1/strided_slice_1/stack_1?
7sequential_1/lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_1/lstm_1/lstm_cell_1/strided_slice_1/stack_2?
/sequential_1/lstm_1/lstm_cell_1/strided_slice_1StridedSlice8sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_1:value:0>sequential_1/lstm_1/lstm_cell_1/strided_slice_1/stack:output:0@sequential_1/lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:0@sequential_1/lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask21
/sequential_1/lstm_1/lstm_cell_1/strided_slice_1?
(sequential_1/lstm_1/lstm_cell_1/MatMul_5MatMul)sequential_1/lstm_1/lstm_cell_1/mul_1:z:08sequential_1/lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2*
(sequential_1/lstm_1/lstm_cell_1/MatMul_5?
%sequential_1/lstm_1/lstm_cell_1/add_1AddV22sequential_1/lstm_1/lstm_cell_1/BiasAdd_1:output:02sequential_1/lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2'
%sequential_1/lstm_1/lstm_cell_1/add_1?
)sequential_1/lstm_1/lstm_cell_1/Sigmoid_1Sigmoid)sequential_1/lstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2+
)sequential_1/lstm_1/lstm_cell_1/Sigmoid_1?
%sequential_1/lstm_1/lstm_cell_1/mul_4Mul-sequential_1/lstm_1/lstm_cell_1/Sigmoid_1:y:0$sequential_1/lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2'
%sequential_1/lstm_1/lstm_cell_1/mul_4?
0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp7sequential_1_lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype022
0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_2?
5sequential_1/lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   27
5sequential_1/lstm_1/lstm_cell_1/strided_slice_2/stack?
7sequential_1/lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   29
7sequential_1/lstm_1/lstm_cell_1/strided_slice_2/stack_1?
7sequential_1/lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_1/lstm_1/lstm_cell_1/strided_slice_2/stack_2?
/sequential_1/lstm_1/lstm_cell_1/strided_slice_2StridedSlice8sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_2:value:0>sequential_1/lstm_1/lstm_cell_1/strided_slice_2/stack:output:0@sequential_1/lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:0@sequential_1/lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask21
/sequential_1/lstm_1/lstm_cell_1/strided_slice_2?
(sequential_1/lstm_1/lstm_cell_1/MatMul_6MatMul)sequential_1/lstm_1/lstm_cell_1/mul_2:z:08sequential_1/lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2*
(sequential_1/lstm_1/lstm_cell_1/MatMul_6?
%sequential_1/lstm_1/lstm_cell_1/add_2AddV22sequential_1/lstm_1/lstm_cell_1/BiasAdd_2:output:02sequential_1/lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2'
%sequential_1/lstm_1/lstm_cell_1/add_2?
$sequential_1/lstm_1/lstm_cell_1/TanhTanh)sequential_1/lstm_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2&
$sequential_1/lstm_1/lstm_cell_1/Tanh?
%sequential_1/lstm_1/lstm_cell_1/mul_5Mul+sequential_1/lstm_1/lstm_cell_1/Sigmoid:y:0(sequential_1/lstm_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2'
%sequential_1/lstm_1/lstm_cell_1/mul_5?
%sequential_1/lstm_1/lstm_cell_1/add_3AddV2)sequential_1/lstm_1/lstm_cell_1/mul_4:z:0)sequential_1/lstm_1/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2'
%sequential_1/lstm_1/lstm_cell_1/add_3?
0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp7sequential_1_lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype022
0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_3?
5sequential_1/lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   27
5sequential_1/lstm_1/lstm_cell_1/strided_slice_3/stack?
7sequential_1/lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7sequential_1/lstm_1/lstm_cell_1/strided_slice_3/stack_1?
7sequential_1/lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_1/lstm_1/lstm_cell_1/strided_slice_3/stack_2?
/sequential_1/lstm_1/lstm_cell_1/strided_slice_3StridedSlice8sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_3:value:0>sequential_1/lstm_1/lstm_cell_1/strided_slice_3/stack:output:0@sequential_1/lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:0@sequential_1/lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask21
/sequential_1/lstm_1/lstm_cell_1/strided_slice_3?
(sequential_1/lstm_1/lstm_cell_1/MatMul_7MatMul)sequential_1/lstm_1/lstm_cell_1/mul_3:z:08sequential_1/lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2*
(sequential_1/lstm_1/lstm_cell_1/MatMul_7?
%sequential_1/lstm_1/lstm_cell_1/add_4AddV22sequential_1/lstm_1/lstm_cell_1/BiasAdd_3:output:02sequential_1/lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2'
%sequential_1/lstm_1/lstm_cell_1/add_4?
)sequential_1/lstm_1/lstm_cell_1/Sigmoid_2Sigmoid)sequential_1/lstm_1/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2+
)sequential_1/lstm_1/lstm_cell_1/Sigmoid_2?
&sequential_1/lstm_1/lstm_cell_1/Tanh_1Tanh)sequential_1/lstm_1/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2(
&sequential_1/lstm_1/lstm_cell_1/Tanh_1?
%sequential_1/lstm_1/lstm_cell_1/mul_6Mul-sequential_1/lstm_1/lstm_cell_1/Sigmoid_2:y:0*sequential_1/lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2'
%sequential_1/lstm_1/lstm_cell_1/mul_6?
1sequential_1/lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1sequential_1/lstm_1/TensorArrayV2_1/element_shape?
#sequential_1/lstm_1/TensorArrayV2_1TensorListReserve:sequential_1/lstm_1/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_1/lstm_1/TensorArrayV2_1v
sequential_1/lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/lstm_1/time?
1sequential_1/lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_1/lstm_1/TensorArrayV2_2/element_shape?
#sequential_1/lstm_1/TensorArrayV2_2TensorListReserve:sequential_1/lstm_1/TensorArrayV2_2/element_shape:output:0,sequential_1/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02%
#sequential_1/lstm_1/TensorArrayV2_2?
Ksequential_1/lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Ksequential_1/lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
=sequential_1/lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor#sequential_1/lstm_1/transpose_1:y:0Tsequential_1/lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02?
=sequential_1/lstm_1/TensorArrayUnstack_1/TensorListFromTensor?
sequential_1/lstm_1/zeros_like	ZerosLike)sequential_1/lstm_1/lstm_cell_1/mul_6:z:0*
T0*'
_output_shapes
:?????????2 
sequential_1/lstm_1/zeros_like?
,sequential_1/lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_1/lstm_1/while/maximum_iterations?
&sequential_1/lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/lstm_1/while/loop_counter?	
sequential_1/lstm_1/whileWhile/sequential_1/lstm_1/while/loop_counter:output:05sequential_1/lstm_1/while/maximum_iterations:output:0!sequential_1/lstm_1/time:output:0,sequential_1/lstm_1/TensorArrayV2_1:handle:0"sequential_1/lstm_1/zeros_like:y:0"sequential_1/lstm_1/zeros:output:0$sequential_1/lstm_1/zeros_1:output:0,sequential_1/lstm_1/strided_slice_1:output:0Ksequential_1/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_1/lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0=sequential_1_lstm_1_lstm_cell_1_split_readvariableop_resource?sequential_1_lstm_1_lstm_cell_1_split_1_readvariableop_resource7sequential_1_lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *%
_read_only_resource_inputs

*3
body+R)
'sequential_1_lstm_1_while_body_14157173*3
cond+R)
'sequential_1_lstm_1_while_cond_14157172*`
output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *
parallel_iterations 2
sequential_1/lstm_1/while?
Dsequential_1/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsequential_1/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_1/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_1/while:output:3Msequential_1/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype028
6sequential_1/lstm_1/TensorArrayV2Stack/TensorListStack?
)sequential_1/lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_1/lstm_1/strided_slice_3/stack?
+sequential_1/lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/lstm_1/strided_slice_3/stack_1?
+sequential_1/lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_1/strided_slice_3/stack_2?
#sequential_1/lstm_1/strided_slice_3StridedSlice?sequential_1/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_1/strided_slice_3/stack:output:04sequential_1/lstm_1/strided_slice_3/stack_1:output:04sequential_1/lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2%
#sequential_1/lstm_1/strided_slice_3?
$sequential_1/lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_1/transpose_2/perm?
sequential_1/lstm_1/transpose_2	Transpose?sequential_1/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2!
sequential_1/lstm_1/transpose_2?
sequential_1/lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/lstm_1/runtime?
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOp?
sequential_1/dense_1/MatMulMatMul,sequential_1/lstm_1/strided_slice_3:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/dense_1/MatMul?
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOp?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/dense_1/BiasAdd?
)sequential_1/lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)sequential_1/lambda_1/strided_slice/stack?
+sequential_1/lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+sequential_1/lambda_1/strided_slice/stack_1?
+sequential_1/lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+sequential_1/lambda_1/strided_slice/stack_2?
#sequential_1/lambda_1/strided_sliceStridedSlice%sequential_1/dense_1/BiasAdd:output:02sequential_1/lambda_1/strided_slice/stack:output:04sequential_1/lambda_1/strided_slice/stack_1:output:04sequential_1/lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2%
#sequential_1/lambda_1/strided_slice?
+sequential_1/lambda_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+sequential_1/lambda_1/strided_slice_1/stack?
-sequential_1/lambda_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-sequential_1/lambda_1/strided_slice_1/stack_1?
-sequential_1/lambda_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-sequential_1/lambda_1/strided_slice_1/stack_2?
%sequential_1/lambda_1/strided_slice_1StridedSlice%sequential_1/dense_1/BiasAdd:output:04sequential_1/lambda_1/strided_slice_1/stack:output:06sequential_1/lambda_1/strided_slice_1/stack_1:output:06sequential_1/lambda_1/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2'
%sequential_1/lambda_1/strided_slice_1
sequential_1/lambda_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential_1/lambda_1/mul/x?
sequential_1/lambda_1/mulMul$sequential_1/lambda_1/mul/x:output:0,sequential_1/lambda_1/strided_slice:output:0*
T0*#
_output_shapes
:?????????2
sequential_1/lambda_1/mul?
sequential_1/lambda_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential_1/lambda_1/mul_1/x?
sequential_1/lambda_1/mul_1Mul&sequential_1/lambda_1/mul_1/x:output:0.sequential_1/lambda_1/strided_slice_1:output:0*
T0*#
_output_shapes
:?????????2
sequential_1/lambda_1/mul_1?
sequential_1/lambda_1/ExpExpsequential_1/lambda_1/mul:z:0*
T0*#
_output_shapes
:?????????2
sequential_1/lambda_1/Exp?
sequential_1/lambda_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?H2
sequential_1/lambda_1/mul_2/x?
sequential_1/lambda_1/mul_2Mul&sequential_1/lambda_1/mul_2/x:output:0sequential_1/lambda_1/Exp:y:0*
T0*#
_output_shapes
:?????????2
sequential_1/lambda_1/mul_2
sequential_1/lambda_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *9?@2
sequential_1/lambda_1/sub/y?
sequential_1/lambda_1/subSubsequential_1/lambda_1/mul_1:z:0$sequential_1/lambda_1/sub/y:output:0*
T0*#
_output_shapes
:?????????2
sequential_1/lambda_1/sub?
sequential_1/lambda_1/SigmoidSigmoidsequential_1/lambda_1/sub:z:0*
T0*#
_output_shapes
:?????????2
sequential_1/lambda_1/Sigmoid?
sequential_1/lambda_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2
sequential_1/lambda_1/mul_3/x?
sequential_1/lambda_1/mul_3Mul&sequential_1/lambda_1/mul_3/x:output:0!sequential_1/lambda_1/Sigmoid:y:0*
T0*#
_output_shapes
:?????????2
sequential_1/lambda_1/mul_3?
sequential_1/lambda_1/stackPacksequential_1/lambda_1/mul_2:z:0sequential_1/lambda_1/mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*
axis?????????2
sequential_1/lambda_1/stack?
IdentityIdentity$sequential_1/lambda_1/stack:output:0<^sequential_1/batch_normalization_1/batchnorm/ReadVariableOp>^sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1>^sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2@^sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp/^sequential_1/lstm_1/lstm_cell_1/ReadVariableOp1^sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_11^sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_21^sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_35^sequential_1/lstm_1/lstm_cell_1/split/ReadVariableOp7^sequential_1/lstm_1/lstm_cell_1/split_1/ReadVariableOp^sequential_1/lstm_1/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:??????????????????:::::::::2z
;sequential_1/batch_normalization_1/batchnorm/ReadVariableOp;sequential_1/batch_normalization_1/batchnorm/ReadVariableOp2~
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_12~
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_22?
?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2`
.sequential_1/lstm_1/lstm_cell_1/ReadVariableOp.sequential_1/lstm_1/lstm_cell_1/ReadVariableOp2d
0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_10sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_12d
0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_20sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_22d
0sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_30sequential_1/lstm_1/lstm_cell_1/ReadVariableOp_32l
4sequential_1/lstm_1/lstm_cell_1/split/ReadVariableOp4sequential_1/lstm_1/lstm_cell_1/split/ReadVariableOp2p
6sequential_1/lstm_1/lstm_cell_1/split_1/ReadVariableOp6sequential_1/lstm_1/lstm_cell_1/split_1/ReadVariableOp26
sequential_1/lstm_1/whilesequential_1/lstm_1/while:e a
4
_output_shapes"
 :??????????????????
)
_user_specified_namemasking_1_input
?
?
while_cond_14162410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_14162410___redundant_placeholder06
2while_while_cond_14162410___redundant_placeholder16
2while_while_cond_14162410___redundant_placeholder26
2while_while_cond_14162410___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?	
/__inference_sequential_1_layer_call_fn_14161509
masking_1_input2
.batch_normalization_1_assignmovingavg_141611294
0batch_normalization_1_assignmovingavg_1_14161135?
;batch_normalization_1_batchnorm_mul_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource4
0lstm_1_lstm_cell_1_split_readvariableop_resource6
2lstm_1_lstm_cell_1_split_1_readvariableop_resource.
*lstm_1_lstm_cell_1_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_1/batchnorm/ReadVariableOp?2batch_normalization_1/batchnorm/mul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?!lstm_1/lstm_cell_1/ReadVariableOp?#lstm_1/lstm_cell_1/ReadVariableOp_1?#lstm_1/lstm_cell_1/ReadVariableOp_2?#lstm_1/lstm_cell_1/ReadVariableOp_3?'lstm_1/lstm_cell_1/split/ReadVariableOp?)lstm_1/lstm_cell_1/split_1/ReadVariableOp?lstm_1/whileq
masking_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
masking_1/NotEqual/y?
masking_1/NotEqualNotEqualmasking_1_inputmasking_1/NotEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
masking_1/NotEqual?
masking_1/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
masking_1/Any/reduction_indices?
masking_1/AnyAnymasking_1/NotEqual:z:0(masking_1/Any/reduction_indices:output:0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
masking_1/Any?
masking_1/CastCastmasking_1/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
masking_1/Cast?
masking_1/mulMulmasking_1_inputmasking_1/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
masking_1/mul?
masking_1/SqueezeSqueezemasking_1/Any:output:0*
T0
*0
_output_shapes
:??????????????????*
squeeze_dims

?????????2
masking_1/Squeeze?
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices?
"batch_normalization_1/moments/meanMeanmasking_1/mul:z:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2$
"batch_normalization_1/moments/mean?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:2,
*batch_normalization_1/moments/StopGradient?
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencemasking_1/mul:z:03batch_normalization_1/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????21
/batch_normalization_1/moments/SquaredDifference?
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indices?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2(
&batch_normalization_1/moments/variance?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1?
+batch_normalization_1/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/14161129*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_1/AssignMovingAvg/decay?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_1_assignmovingavg_14161129*
_output_shapes
:*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/14161129*
_output_shapes
:2+
)batch_normalization_1/AssignMovingAvg/sub?
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/14161129*
_output_shapes
:2+
)batch_normalization_1/AssignMovingAvg/mul?
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_1_assignmovingavg_14161129-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/14161129*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_1/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/14161135*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_1/AssignMovingAvg_1/decay?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_1_assignmovingavg_1_14161135*
_output_shapes
:*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/14161135*
_output_shapes
:2-
+batch_normalization_1/AssignMovingAvg_1/sub?
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/14161135*
_output_shapes
:2-
+batch_normalization_1/AssignMovingAvg_1/mul?
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_1_assignmovingavg_1_14161135/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/14161135*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/Rsqrt?
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Mulmasking_1/mul:z:0'batch_normalization_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2'
%batch_normalization_1/batchnorm/mul_1?
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/mul_2?
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp?
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2'
%batch_normalization_1/batchnorm/add_1u
lstm_1/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
lstm_1/Shape?
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice/stack?
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_1?
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_2?
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slicej
lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros/mul/y?
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/mulm
lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros/Less/y?
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/Lessp
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros/packed/1?
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros/packedm
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros/Const?
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/zerosn
lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros_1/mul/y?
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/mulq
lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros_1/Less/y?
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/Lesst
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros_1/packed/1?
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros_1/packedq
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros_1/Const?
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/zeros_1?
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose/perm?
lstm_1/transpose	Transpose)batch_normalization_1/batchnorm/add_1:z:0lstm_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_1/transposed
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:2
lstm_1/Shape_1?
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_1/stack?
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_1?
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_2?
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slice_1y
lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_1/ExpandDims/dim?
lstm_1/ExpandDims
ExpandDimsmasking_1/Squeeze:output:0lstm_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????2
lstm_1/ExpandDims?
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_1/perm?
lstm_1/transpose_1	Transposelstm_1/ExpandDims:output:0 lstm_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????2
lstm_1/transpose_1?
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_1/TensorArrayV2/element_shape?
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2?
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_1/TensorArrayUnstack/TensorListFromTensor?
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_2/stack?
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_1?
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_2?
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_1/strided_slice_2?
"lstm_1/lstm_cell_1/ones_like/ShapeShapelstm_1/zeros:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/ones_like/Shape?
"lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"lstm_1/lstm_cell_1/ones_like/Const?
lstm_1/lstm_cell_1/ones_likeFill+lstm_1/lstm_cell_1/ones_like/Shape:output:0+lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/ones_like?
 lstm_1/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2"
 lstm_1/lstm_cell_1/dropout/Const?
lstm_1/lstm_cell_1/dropout/MulMul%lstm_1/lstm_cell_1/ones_like:output:0)lstm_1/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2 
lstm_1/lstm_cell_1/dropout/Mul?
 lstm_1/lstm_cell_1/dropout/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell_1/dropout/Shape?
7lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??"29
7lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniform?
)lstm_1/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2+
)lstm_1/lstm_cell_1/dropout/GreaterEqual/y?
'lstm_1/lstm_cell_1/dropout/GreaterEqualGreaterEqual@lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniform:output:02lstm_1/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2)
'lstm_1/lstm_cell_1/dropout/GreaterEqual?
lstm_1/lstm_cell_1/dropout/CastCast+lstm_1/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2!
lstm_1/lstm_cell_1/dropout/Cast?
 lstm_1/lstm_cell_1/dropout/Mul_1Mul"lstm_1/lstm_cell_1/dropout/Mul:z:0#lstm_1/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/lstm_cell_1/dropout/Mul_1?
"lstm_1/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2$
"lstm_1/lstm_cell_1/dropout_1/Const?
 lstm_1/lstm_cell_1/dropout_1/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/lstm_cell_1/dropout_1/Mul?
"lstm_1/lstm_cell_1/dropout_1/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/dropout_1/Shape?
9lstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ә?2;
9lstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
+lstm_1/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2-
+lstm_1/lstm_cell_1/dropout_1/GreaterEqual/y?
)lstm_1/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2+
)lstm_1/lstm_cell_1/dropout_1/GreaterEqual?
!lstm_1/lstm_cell_1/dropout_1/CastCast-lstm_1/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2#
!lstm_1/lstm_cell_1/dropout_1/Cast?
"lstm_1/lstm_cell_1/dropout_1/Mul_1Mul$lstm_1/lstm_cell_1/dropout_1/Mul:z:0%lstm_1/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/lstm_cell_1/dropout_1/Mul_1?
"lstm_1/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2$
"lstm_1/lstm_cell_1/dropout_2/Const?
 lstm_1/lstm_cell_1/dropout_2/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/lstm_cell_1/dropout_2/Mul?
"lstm_1/lstm_cell_1/dropout_2/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/dropout_2/Shape?
9lstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2?˛2;
9lstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
+lstm_1/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2-
+lstm_1/lstm_cell_1/dropout_2/GreaterEqual/y?
)lstm_1/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2+
)lstm_1/lstm_cell_1/dropout_2/GreaterEqual?
!lstm_1/lstm_cell_1/dropout_2/CastCast-lstm_1/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2#
!lstm_1/lstm_cell_1/dropout_2/Cast?
"lstm_1/lstm_cell_1/dropout_2/Mul_1Mul$lstm_1/lstm_cell_1/dropout_2/Mul:z:0%lstm_1/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/lstm_cell_1/dropout_2/Mul_1?
"lstm_1/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2$
"lstm_1/lstm_cell_1/dropout_3/Const?
 lstm_1/lstm_cell_1/dropout_3/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/lstm_cell_1/dropout_3/Mul?
"lstm_1/lstm_cell_1/dropout_3/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/dropout_3/Shape?
9lstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2?ף2;
9lstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
+lstm_1/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2-
+lstm_1/lstm_cell_1/dropout_3/GreaterEqual/y?
)lstm_1/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2+
)lstm_1/lstm_cell_1/dropout_3/GreaterEqual?
!lstm_1/lstm_cell_1/dropout_3/CastCast-lstm_1/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2#
!lstm_1/lstm_cell_1/dropout_3/Cast?
"lstm_1/lstm_cell_1/dropout_3/Mul_1Mul$lstm_1/lstm_cell_1/dropout_3/Mul:z:0%lstm_1/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/lstm_cell_1/dropout_3/Mul_1v
lstm_1/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const?
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_1/lstm_cell_1/split/split_dim?
'lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02)
'lstm_1/lstm_cell_1/split/ReadVariableOp?
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_1/lstm_cell_1/split?
lstm_1/lstm_cell_1/MatMulMatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul?
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_1?
lstm_1/lstm_cell_1/MatMul_2MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_2?
lstm_1/lstm_cell_1/MatMul_3MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_3z
lstm_1/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const_1?
$lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_1/lstm_cell_1/split_1/split_dim?
)lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02+
)lstm_1/lstm_cell_1/split_1/ReadVariableOp?
lstm_1/lstm_cell_1/split_1Split-lstm_1/lstm_cell_1/split_1/split_dim:output:01lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_1/lstm_cell_1/split_1?
lstm_1/lstm_cell_1/BiasAddBiasAdd#lstm_1/lstm_cell_1/MatMul:product:0#lstm_1/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd?
lstm_1/lstm_cell_1/BiasAdd_1BiasAdd%lstm_1/lstm_cell_1/MatMul_1:product:0#lstm_1/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_1?
lstm_1/lstm_cell_1/BiasAdd_2BiasAdd%lstm_1/lstm_cell_1/MatMul_2:product:0#lstm_1/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_2?
lstm_1/lstm_cell_1/BiasAdd_3BiasAdd%lstm_1/lstm_cell_1/MatMul_3:product:0#lstm_1/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_3?
lstm_1/lstm_cell_1/mulMullstm_1/zeros:output:0$lstm_1/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul?
lstm_1/lstm_cell_1/mul_1Mullstm_1/zeros:output:0&lstm_1/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_1?
lstm_1/lstm_cell_1/mul_2Mullstm_1/zeros:output:0&lstm_1/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_2?
lstm_1/lstm_cell_1/mul_3Mullstm_1/zeros:output:0&lstm_1/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_3?
!lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02#
!lstm_1/lstm_cell_1/ReadVariableOp?
&lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_1/lstm_cell_1/strided_slice/stack?
(lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice/stack_1?
(lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell_1/strided_slice/stack_2?
 lstm_1/lstm_cell_1/strided_sliceStridedSlice)lstm_1/lstm_cell_1/ReadVariableOp:value:0/lstm_1/lstm_cell_1/strided_slice/stack:output:01lstm_1/lstm_cell_1/strided_slice/stack_1:output:01lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2"
 lstm_1/lstm_cell_1/strided_slice?
lstm_1/lstm_cell_1/MatMul_4MatMullstm_1/lstm_cell_1/mul:z:0)lstm_1/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_4?
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/BiasAdd:output:0%lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add?
lstm_1/lstm_cell_1/SigmoidSigmoidlstm_1/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid?
#lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_1?
(lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice_1/stack?
*lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_1?
*lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_2?
"lstm_1/lstm_cell_1/strided_slice_1StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_1:value:01lstm_1/lstm_cell_1/strided_slice_1/stack:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_1?
lstm_1/lstm_cell_1/MatMul_5MatMullstm_1/lstm_cell_1/mul_1:z:0+lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_5?
lstm_1/lstm_cell_1/add_1AddV2%lstm_1/lstm_cell_1/BiasAdd_1:output:0%lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_1?
lstm_1/lstm_cell_1/Sigmoid_1Sigmoidlstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid_1?
lstm_1/lstm_cell_1/mul_4Mul lstm_1/lstm_cell_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_4?
#lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_2?
(lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2*
(lstm_1/lstm_cell_1/strided_slice_2/stack?
*lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_1?
*lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_2?
"lstm_1/lstm_cell_1/strided_slice_2StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_2:value:01lstm_1/lstm_cell_1/strided_slice_2/stack:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_2?
lstm_1/lstm_cell_1/MatMul_6MatMullstm_1/lstm_cell_1/mul_2:z:0+lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_6?
lstm_1/lstm_cell_1/add_2AddV2%lstm_1/lstm_cell_1/BiasAdd_2:output:0%lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_2?
lstm_1/lstm_cell_1/TanhTanhlstm_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Tanh?
lstm_1/lstm_cell_1/mul_5Mullstm_1/lstm_cell_1/Sigmoid:y:0lstm_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_5?
lstm_1/lstm_cell_1/add_3AddV2lstm_1/lstm_cell_1/mul_4:z:0lstm_1/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_3?
#lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_3?
(lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2*
(lstm_1/lstm_cell_1/strided_slice_3/stack?
*lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_1?
*lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_2?
"lstm_1/lstm_cell_1/strided_slice_3StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_3:value:01lstm_1/lstm_cell_1/strided_slice_3/stack:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_3?
lstm_1/lstm_cell_1/MatMul_7MatMullstm_1/lstm_cell_1/mul_3:z:0+lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_7?
lstm_1/lstm_cell_1/add_4AddV2%lstm_1/lstm_cell_1/BiasAdd_3:output:0%lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_4?
lstm_1/lstm_cell_1/Sigmoid_2Sigmoidlstm_1/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid_2?
lstm_1/lstm_cell_1/Tanh_1Tanhlstm_1/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Tanh_1?
lstm_1/lstm_cell_1/mul_6Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_6?
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$lstm_1/TensorArrayV2_1/element_shape?
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2_1\
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/time?
$lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$lstm_1/TensorArrayV2_2/element_shape?
lstm_1/TensorArrayV2_2TensorListReserve-lstm_1/TensorArrayV2_2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
lstm_1/TensorArrayV2_2?
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
0lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm_1/transpose_1:y:0Glstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type022
0lstm_1/TensorArrayUnstack_1/TensorListFromTensor?
lstm_1/zeros_like	ZerosLikelstm_1/lstm_cell_1/mul_6:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/zeros_like?
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_1/while/maximum_iterationsx
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/while/loop_counter?
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros_like:y:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:00lstm_1_lstm_cell_1_split_readvariableop_resource2lstm_1_lstm_cell_1_split_1_readvariableop_resource*lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *%
_read_only_resource_inputs

*&
bodyR
lstm_1_while_body_14161303*&
condR
lstm_1_while_cond_14161302*`
output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *
parallel_iterations 2
lstm_1/while?
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02+
)lstm_1/TensorArrayV2Stack/TensorListStack?
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_1/strided_slice_3/stack?
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_1/strided_slice_3/stack_1?
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_3/stack_2?
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_1/strided_slice_3?
lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_2/perm?
lstm_1/transpose_2	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_1/transpose_2t
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/runtime?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMullstm_1/strided_slice_3:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lambda_1/strided_slice/stack?
lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_1/strided_slice/stack_1?
lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_1/strided_slice/stack_2?
lambda_1/strided_sliceStridedSlicedense_1/BiasAdd:output:0%lambda_1/strided_slice/stack:output:0'lambda_1/strided_slice/stack_1:output:0'lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
lambda_1/strided_slice?
lambda_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_1/strided_slice_1/stack?
 lambda_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 lambda_1/strided_slice_1/stack_1?
 lambda_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 lambda_1/strided_slice_1/stack_2?
lambda_1/strided_slice_1StridedSlicedense_1/BiasAdd:output:0'lambda_1/strided_slice_1/stack:output:0)lambda_1/strided_slice_1/stack_1:output:0)lambda_1/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
lambda_1/strided_slice_1e
lambda_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda_1/mul/x?
lambda_1/mulMullambda_1/mul/x:output:0lambda_1/strided_slice:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/muli
lambda_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda_1/mul_1/x?
lambda_1/mul_1Mullambda_1/mul_1/x:output:0!lambda_1/strided_slice_1:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_1c
lambda_1/ExpExplambda_1/mul:z:0*
T0*#
_output_shapes
:?????????2
lambda_1/Expi
lambda_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?H2
lambda_1/mul_2/x?
lambda_1/mul_2Mullambda_1/mul_2/x:output:0lambda_1/Exp:y:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_2e
lambda_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *9?@2
lambda_1/sub/y~
lambda_1/subSublambda_1/mul_1:z:0lambda_1/sub/y:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/subo
lambda_1/SigmoidSigmoidlambda_1/sub:z:0*
T0*#
_output_shapes
:?????????2
lambda_1/Sigmoidi
lambda_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2
lambda_1/mul_3/x?
lambda_1/mul_3Mullambda_1/mul_3/x:output:0lambda_1/Sigmoid:y:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_3?
lambda_1/stackPacklambda_1/mul_2:z:0lambda_1/mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*
axis?????????2
lambda_1/stack?
IdentityIdentitylambda_1/stack:output:0:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:??????????????????:::::::::2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2F
!lstm_1/lstm_cell_1/ReadVariableOp!lstm_1/lstm_cell_1/ReadVariableOp2J
#lstm_1/lstm_cell_1/ReadVariableOp_1#lstm_1/lstm_cell_1/ReadVariableOp_12J
#lstm_1/lstm_cell_1/ReadVariableOp_2#lstm_1/lstm_cell_1/ReadVariableOp_22J
#lstm_1/lstm_cell_1/ReadVariableOp_3#lstm_1/lstm_cell_1/ReadVariableOp_32R
'lstm_1/lstm_cell_1/split/ReadVariableOp'lstm_1/lstm_cell_1/split/ReadVariableOp2V
)lstm_1/lstm_cell_1/split_1/ReadVariableOp)lstm_1/lstm_cell_1/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:e a
4
_output_shapes"
 :??????????????????
)
_user_specified_namemasking_1_input
??
?	
J__inference_sequential_1_layer_call_and_return_conditional_losses_14160792
masking_1_input2
.batch_normalization_1_assignmovingavg_141590844
0batch_normalization_1_assignmovingavg_1_14159090?
;batch_normalization_1_batchnorm_mul_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource4
0lstm_1_lstm_cell_1_split_readvariableop_resource6
2lstm_1_lstm_cell_1_split_1_readvariableop_resource.
*lstm_1_lstm_cell_1_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_1/batchnorm/ReadVariableOp?2batch_normalization_1/batchnorm/mul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?!lstm_1/lstm_cell_1/ReadVariableOp?#lstm_1/lstm_cell_1/ReadVariableOp_1?#lstm_1/lstm_cell_1/ReadVariableOp_2?#lstm_1/lstm_cell_1/ReadVariableOp_3?'lstm_1/lstm_cell_1/split/ReadVariableOp?)lstm_1/lstm_cell_1/split_1/ReadVariableOp?lstm_1/whileq
masking_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
masking_1/NotEqual/y?
masking_1/NotEqualNotEqualmasking_1_inputmasking_1/NotEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
masking_1/NotEqual?
masking_1/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
masking_1/Any/reduction_indices?
masking_1/AnyAnymasking_1/NotEqual:z:0(masking_1/Any/reduction_indices:output:0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
masking_1/Any?
masking_1/CastCastmasking_1/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
masking_1/Cast?
masking_1/mulMulmasking_1_inputmasking_1/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
masking_1/mul?
masking_1/SqueezeSqueezemasking_1/Any:output:0*
T0
*0
_output_shapes
:??????????????????*
squeeze_dims

?????????2
masking_1/Squeeze?
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices?
"batch_normalization_1/moments/meanMeanmasking_1/mul:z:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2$
"batch_normalization_1/moments/mean?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:2,
*batch_normalization_1/moments/StopGradient?
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencemasking_1/mul:z:03batch_normalization_1/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????21
/batch_normalization_1/moments/SquaredDifference?
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indices?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2(
&batch_normalization_1/moments/variance?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1?
+batch_normalization_1/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/14159084*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_1/AssignMovingAvg/decay?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_1_assignmovingavg_14159084*
_output_shapes
:*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/14159084*
_output_shapes
:2+
)batch_normalization_1/AssignMovingAvg/sub?
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/14159084*
_output_shapes
:2+
)batch_normalization_1/AssignMovingAvg/mul?
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_1_assignmovingavg_14159084-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/14159084*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_1/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/14159090*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_1/AssignMovingAvg_1/decay?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_1_assignmovingavg_1_14159090*
_output_shapes
:*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/14159090*
_output_shapes
:2-
+batch_normalization_1/AssignMovingAvg_1/sub?
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/14159090*
_output_shapes
:2-
+batch_normalization_1/AssignMovingAvg_1/mul?
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_1_assignmovingavg_1_14159090/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/14159090*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/Rsqrt?
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Mulmasking_1/mul:z:0'batch_normalization_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2'
%batch_normalization_1/batchnorm/mul_1?
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/mul_2?
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp?
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2'
%batch_normalization_1/batchnorm/add_1u
lstm_1/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
lstm_1/Shape?
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice/stack?
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_1?
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_2?
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slicej
lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros/mul/y?
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/mulm
lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros/Less/y?
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/Lessp
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros/packed/1?
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros/packedm
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros/Const?
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/zerosn
lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros_1/mul/y?
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/mulq
lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros_1/Less/y?
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/Lesst
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros_1/packed/1?
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros_1/packedq
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros_1/Const?
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/zeros_1?
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose/perm?
lstm_1/transpose	Transpose)batch_normalization_1/batchnorm/add_1:z:0lstm_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_1/transposed
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:2
lstm_1/Shape_1?
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_1/stack?
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_1?
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_2?
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slice_1y
lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_1/ExpandDims/dim?
lstm_1/ExpandDims
ExpandDimsmasking_1/Squeeze:output:0lstm_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????2
lstm_1/ExpandDims?
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_1/perm?
lstm_1/transpose_1	Transposelstm_1/ExpandDims:output:0 lstm_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????2
lstm_1/transpose_1?
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_1/TensorArrayV2/element_shape?
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2?
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_1/TensorArrayUnstack/TensorListFromTensor?
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_2/stack?
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_1?
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_2?
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_1/strided_slice_2?
"lstm_1/lstm_cell_1/ones_like/ShapeShapelstm_1/zeros:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/ones_like/Shape?
"lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"lstm_1/lstm_cell_1/ones_like/Const?
lstm_1/lstm_cell_1/ones_likeFill+lstm_1/lstm_cell_1/ones_like/Shape:output:0+lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/ones_like?
 lstm_1/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2"
 lstm_1/lstm_cell_1/dropout/Const?
lstm_1/lstm_cell_1/dropout/MulMul%lstm_1/lstm_cell_1/ones_like:output:0)lstm_1/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2 
lstm_1/lstm_cell_1/dropout/Mul?
 lstm_1/lstm_cell_1/dropout/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell_1/dropout/Shape?
7lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??429
7lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniform?
)lstm_1/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2+
)lstm_1/lstm_cell_1/dropout/GreaterEqual/y?
'lstm_1/lstm_cell_1/dropout/GreaterEqualGreaterEqual@lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniform:output:02lstm_1/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2)
'lstm_1/lstm_cell_1/dropout/GreaterEqual?
lstm_1/lstm_cell_1/dropout/CastCast+lstm_1/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2!
lstm_1/lstm_cell_1/dropout/Cast?
 lstm_1/lstm_cell_1/dropout/Mul_1Mul"lstm_1/lstm_cell_1/dropout/Mul:z:0#lstm_1/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/lstm_cell_1/dropout/Mul_1?
"lstm_1/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2$
"lstm_1/lstm_cell_1/dropout_1/Const?
 lstm_1/lstm_cell_1/dropout_1/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/lstm_cell_1/dropout_1/Mul?
"lstm_1/lstm_cell_1/dropout_1/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/dropout_1/Shape?
9lstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2;
9lstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
+lstm_1/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2-
+lstm_1/lstm_cell_1/dropout_1/GreaterEqual/y?
)lstm_1/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2+
)lstm_1/lstm_cell_1/dropout_1/GreaterEqual?
!lstm_1/lstm_cell_1/dropout_1/CastCast-lstm_1/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2#
!lstm_1/lstm_cell_1/dropout_1/Cast?
"lstm_1/lstm_cell_1/dropout_1/Mul_1Mul$lstm_1/lstm_cell_1/dropout_1/Mul:z:0%lstm_1/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/lstm_cell_1/dropout_1/Mul_1?
"lstm_1/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2$
"lstm_1/lstm_cell_1/dropout_2/Const?
 lstm_1/lstm_cell_1/dropout_2/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/lstm_cell_1/dropout_2/Mul?
"lstm_1/lstm_cell_1/dropout_2/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/dropout_2/Shape?
9lstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2;
9lstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
+lstm_1/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2-
+lstm_1/lstm_cell_1/dropout_2/GreaterEqual/y?
)lstm_1/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2+
)lstm_1/lstm_cell_1/dropout_2/GreaterEqual?
!lstm_1/lstm_cell_1/dropout_2/CastCast-lstm_1/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2#
!lstm_1/lstm_cell_1/dropout_2/Cast?
"lstm_1/lstm_cell_1/dropout_2/Mul_1Mul$lstm_1/lstm_cell_1/dropout_2/Mul:z:0%lstm_1/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/lstm_cell_1/dropout_2/Mul_1?
"lstm_1/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2$
"lstm_1/lstm_cell_1/dropout_3/Const?
 lstm_1/lstm_cell_1/dropout_3/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/lstm_cell_1/dropout_3/Mul?
"lstm_1/lstm_cell_1/dropout_3/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/dropout_3/Shape?
9lstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2?ޢ2;
9lstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
+lstm_1/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2-
+lstm_1/lstm_cell_1/dropout_3/GreaterEqual/y?
)lstm_1/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2+
)lstm_1/lstm_cell_1/dropout_3/GreaterEqual?
!lstm_1/lstm_cell_1/dropout_3/CastCast-lstm_1/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2#
!lstm_1/lstm_cell_1/dropout_3/Cast?
"lstm_1/lstm_cell_1/dropout_3/Mul_1Mul$lstm_1/lstm_cell_1/dropout_3/Mul:z:0%lstm_1/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/lstm_cell_1/dropout_3/Mul_1v
lstm_1/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const?
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_1/lstm_cell_1/split/split_dim?
'lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02)
'lstm_1/lstm_cell_1/split/ReadVariableOp?
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_1/lstm_cell_1/split?
lstm_1/lstm_cell_1/MatMulMatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul?
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_1?
lstm_1/lstm_cell_1/MatMul_2MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_2?
lstm_1/lstm_cell_1/MatMul_3MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_3z
lstm_1/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const_1?
$lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_1/lstm_cell_1/split_1/split_dim?
)lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02+
)lstm_1/lstm_cell_1/split_1/ReadVariableOp?
lstm_1/lstm_cell_1/split_1Split-lstm_1/lstm_cell_1/split_1/split_dim:output:01lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_1/lstm_cell_1/split_1?
lstm_1/lstm_cell_1/BiasAddBiasAdd#lstm_1/lstm_cell_1/MatMul:product:0#lstm_1/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd?
lstm_1/lstm_cell_1/BiasAdd_1BiasAdd%lstm_1/lstm_cell_1/MatMul_1:product:0#lstm_1/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_1?
lstm_1/lstm_cell_1/BiasAdd_2BiasAdd%lstm_1/lstm_cell_1/MatMul_2:product:0#lstm_1/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_2?
lstm_1/lstm_cell_1/BiasAdd_3BiasAdd%lstm_1/lstm_cell_1/MatMul_3:product:0#lstm_1/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_3?
lstm_1/lstm_cell_1/mulMullstm_1/zeros:output:0$lstm_1/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul?
lstm_1/lstm_cell_1/mul_1Mullstm_1/zeros:output:0&lstm_1/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_1?
lstm_1/lstm_cell_1/mul_2Mullstm_1/zeros:output:0&lstm_1/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_2?
lstm_1/lstm_cell_1/mul_3Mullstm_1/zeros:output:0&lstm_1/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_3?
!lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02#
!lstm_1/lstm_cell_1/ReadVariableOp?
&lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_1/lstm_cell_1/strided_slice/stack?
(lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice/stack_1?
(lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell_1/strided_slice/stack_2?
 lstm_1/lstm_cell_1/strided_sliceStridedSlice)lstm_1/lstm_cell_1/ReadVariableOp:value:0/lstm_1/lstm_cell_1/strided_slice/stack:output:01lstm_1/lstm_cell_1/strided_slice/stack_1:output:01lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2"
 lstm_1/lstm_cell_1/strided_slice?
lstm_1/lstm_cell_1/MatMul_4MatMullstm_1/lstm_cell_1/mul:z:0)lstm_1/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_4?
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/BiasAdd:output:0%lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add?
lstm_1/lstm_cell_1/SigmoidSigmoidlstm_1/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid?
#lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_1?
(lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice_1/stack?
*lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_1?
*lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_2?
"lstm_1/lstm_cell_1/strided_slice_1StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_1:value:01lstm_1/lstm_cell_1/strided_slice_1/stack:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_1?
lstm_1/lstm_cell_1/MatMul_5MatMullstm_1/lstm_cell_1/mul_1:z:0+lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_5?
lstm_1/lstm_cell_1/add_1AddV2%lstm_1/lstm_cell_1/BiasAdd_1:output:0%lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_1?
lstm_1/lstm_cell_1/Sigmoid_1Sigmoidlstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid_1?
lstm_1/lstm_cell_1/mul_4Mul lstm_1/lstm_cell_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_4?
#lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_2?
(lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2*
(lstm_1/lstm_cell_1/strided_slice_2/stack?
*lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_1?
*lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_2?
"lstm_1/lstm_cell_1/strided_slice_2StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_2:value:01lstm_1/lstm_cell_1/strided_slice_2/stack:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_2?
lstm_1/lstm_cell_1/MatMul_6MatMullstm_1/lstm_cell_1/mul_2:z:0+lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_6?
lstm_1/lstm_cell_1/add_2AddV2%lstm_1/lstm_cell_1/BiasAdd_2:output:0%lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_2?
lstm_1/lstm_cell_1/TanhTanhlstm_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Tanh?
lstm_1/lstm_cell_1/mul_5Mullstm_1/lstm_cell_1/Sigmoid:y:0lstm_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_5?
lstm_1/lstm_cell_1/add_3AddV2lstm_1/lstm_cell_1/mul_4:z:0lstm_1/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_3?
#lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_3?
(lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2*
(lstm_1/lstm_cell_1/strided_slice_3/stack?
*lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_1?
*lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_2?
"lstm_1/lstm_cell_1/strided_slice_3StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_3:value:01lstm_1/lstm_cell_1/strided_slice_3/stack:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_3?
lstm_1/lstm_cell_1/MatMul_7MatMullstm_1/lstm_cell_1/mul_3:z:0+lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_7?
lstm_1/lstm_cell_1/add_4AddV2%lstm_1/lstm_cell_1/BiasAdd_3:output:0%lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_4?
lstm_1/lstm_cell_1/Sigmoid_2Sigmoidlstm_1/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid_2?
lstm_1/lstm_cell_1/Tanh_1Tanhlstm_1/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Tanh_1?
lstm_1/lstm_cell_1/mul_6Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_6?
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$lstm_1/TensorArrayV2_1/element_shape?
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2_1\
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/time?
$lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$lstm_1/TensorArrayV2_2/element_shape?
lstm_1/TensorArrayV2_2TensorListReserve-lstm_1/TensorArrayV2_2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
lstm_1/TensorArrayV2_2?
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
0lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm_1/transpose_1:y:0Glstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type022
0lstm_1/TensorArrayUnstack_1/TensorListFromTensor?
lstm_1/zeros_like	ZerosLikelstm_1/lstm_cell_1/mul_6:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/zeros_like?
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_1/while/maximum_iterationsx
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/while/loop_counter?
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros_like:y:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:00lstm_1_lstm_cell_1_split_readvariableop_resource2lstm_1_lstm_cell_1_split_1_readvariableop_resource*lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *%
_read_only_resource_inputs

*&
bodyR
lstm_1_while_body_14160462*&
condR
lstm_1_while_cond_14160461*`
output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *
parallel_iterations 2
lstm_1/while?
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02+
)lstm_1/TensorArrayV2Stack/TensorListStack?
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_1/strided_slice_3/stack?
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_1/strided_slice_3/stack_1?
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_3/stack_2?
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_1/strided_slice_3?
lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_2/perm?
lstm_1/transpose_2	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_1/transpose_2t
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/runtime?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMullstm_1/strided_slice_3:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lambda_1/strided_slice/stack?
lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_1/strided_slice/stack_1?
lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_1/strided_slice/stack_2?
lambda_1/strided_sliceStridedSlicedense_1/BiasAdd:output:0%lambda_1/strided_slice/stack:output:0'lambda_1/strided_slice/stack_1:output:0'lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
lambda_1/strided_slice?
lambda_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_1/strided_slice_1/stack?
 lambda_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 lambda_1/strided_slice_1/stack_1?
 lambda_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 lambda_1/strided_slice_1/stack_2?
lambda_1/strided_slice_1StridedSlicedense_1/BiasAdd:output:0'lambda_1/strided_slice_1/stack:output:0)lambda_1/strided_slice_1/stack_1:output:0)lambda_1/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
lambda_1/strided_slice_1e
lambda_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda_1/mul/x?
lambda_1/mulMullambda_1/mul/x:output:0lambda_1/strided_slice:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/muli
lambda_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda_1/mul_1/x?
lambda_1/mul_1Mullambda_1/mul_1/x:output:0!lambda_1/strided_slice_1:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_1c
lambda_1/ExpExplambda_1/mul:z:0*
T0*#
_output_shapes
:?????????2
lambda_1/Expi
lambda_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?H2
lambda_1/mul_2/x?
lambda_1/mul_2Mullambda_1/mul_2/x:output:0lambda_1/Exp:y:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_2e
lambda_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *9?@2
lambda_1/sub/y~
lambda_1/subSublambda_1/mul_1:z:0lambda_1/sub/y:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/subo
lambda_1/SigmoidSigmoidlambda_1/sub:z:0*
T0*#
_output_shapes
:?????????2
lambda_1/Sigmoidi
lambda_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2
lambda_1/mul_3/x?
lambda_1/mul_3Mullambda_1/mul_3/x:output:0lambda_1/Sigmoid:y:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_3?
lambda_1/stackPacklambda_1/mul_2:z:0lambda_1/mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*
axis?????????2
lambda_1/stack?
IdentityIdentitylambda_1/stack:output:0:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:??????????????????:::::::::2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2F
!lstm_1/lstm_cell_1/ReadVariableOp!lstm_1/lstm_cell_1/ReadVariableOp2J
#lstm_1/lstm_cell_1/ReadVariableOp_1#lstm_1/lstm_cell_1/ReadVariableOp_12J
#lstm_1/lstm_cell_1/ReadVariableOp_2#lstm_1/lstm_cell_1/ReadVariableOp_22J
#lstm_1/lstm_cell_1/ReadVariableOp_3#lstm_1/lstm_cell_1/ReadVariableOp_32R
'lstm_1/lstm_cell_1/split/ReadVariableOp'lstm_1/lstm_cell_1/split/ReadVariableOp2V
)lstm_1/lstm_cell_1/split_1/ReadVariableOp)lstm_1/lstm_cell_1/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:e a
4
_output_shapes"
 :??????????????????
)
_user_specified_namemasking_1_input
?
G
+__inference_lambda_1_layer_call_fn_14164410

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_1S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xg
mulMulmul/x:output:0strided_slice:output:0*
T0*#
_output_shapes
:?????????2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/xo
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*#
_output_shapes
:?????????2
mul_1H
ExpExpmul:z:0*
T0*#
_output_shapes
:?????????2
ExpW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?H2	
mul_2/x^
mul_2Mulmul_2/x:output:0Exp:y:0*
T0*#
_output_shapes
:?????????2
mul_2S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *9?@2
sub/yZ
subSub	mul_1:z:0sub/y:output:0*
T0*#
_output_shapes
:?????????2
subT
SigmoidSigmoidsub:z:0*
T0*#
_output_shapes
:?????????2	
SigmoidW
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2	
mul_3/xb
mul_3Mulmul_3/x:output:0Sigmoid:y:0*
T0*#
_output_shapes
:?????????2
mul_3|
stackPack	mul_2:z:0	mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*
axis?????????2
stackb
IdentityIdentitystack:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
while_cond_14164142
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_16
2while_while_cond_14164142___redundant_placeholder06
2while_while_cond_14164142___redundant_placeholder16
2while_while_cond_14164142___redundant_placeholder26
2while_while_cond_14164142___redundant_placeholder36
2while_while_cond_14164142___redundant_placeholder4
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*j
_input_shapesY
W: : : : :?????????:?????????:?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
??
?
lstm_1_while_body_14161303*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0i
elstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0<
8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0>
:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_06
2lstm_1_while_lstm_cell_1_readvariableop_resource_0
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5
lstm_1_while_identity_6'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorg
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor:
6lstm_1_while_lstm_cell_1_split_readvariableop_resource<
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource4
0lstm_1_while_lstm_cell_1_readvariableop_resource??'lstm_1/while/lstm_cell_1/ReadVariableOp?)lstm_1/while/lstm_cell_1/ReadVariableOp_1?)lstm_1/while/lstm_cell_1/ReadVariableOp_2?)lstm_1/while/lstm_cell_1/ReadVariableOp_3?-lstm_1/while/lstm_cell_1/split/ReadVariableOp?/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0lstm_1/while/TensorArrayV2Read/TensorListGetItem?
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0lstm_1_while_placeholderIlstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
24
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItem?
(lstm_1/while/lstm_cell_1/ones_like/ShapeShapelstm_1_while_placeholder_3*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/ones_like/Shape?
(lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(lstm_1/while/lstm_cell_1/ones_like/Const?
"lstm_1/while/lstm_cell_1/ones_likeFill1lstm_1/while/lstm_cell_1/ones_like/Shape:output:01lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/ones_like?
&lstm_1/while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&lstm_1/while/lstm_cell_1/dropout/Const?
$lstm_1/while/lstm_cell_1/dropout/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:0/lstm_1/while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2&
$lstm_1/while/lstm_cell_1/dropout/Mul?
&lstm_1/while/lstm_cell_1/dropout/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell_1/dropout/Shape?
=lstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2?
=lstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniform?
/lstm_1/while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>21
/lstm_1/while/lstm_cell_1/dropout/GreaterEqual/y?
-lstm_1/while/lstm_cell_1/dropout/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2/
-lstm_1/while/lstm_cell_1/dropout/GreaterEqual?
%lstm_1/while/lstm_cell_1/dropout/CastCast1lstm_1/while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2'
%lstm_1/while/lstm_cell_1/dropout/Cast?
&lstm_1/while/lstm_cell_1/dropout/Mul_1Mul(lstm_1/while/lstm_cell_1/dropout/Mul:z:0)lstm_1/while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2(
&lstm_1/while/lstm_cell_1/dropout/Mul_1?
(lstm_1/while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2*
(lstm_1/while/lstm_cell_1/dropout_1/Const?
&lstm_1/while/lstm_cell_1/dropout_1/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2(
&lstm_1/while/lstm_cell_1/dropout_1/Mul?
(lstm_1/while/lstm_cell_1/dropout_1/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/dropout_1/Shape?
?lstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??#2A
?lstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
1lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>23
1lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/y?
/lstm_1/while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????21
/lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual?
'lstm_1/while/lstm_cell_1/dropout_1/CastCast3lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2)
'lstm_1/while/lstm_cell_1/dropout_1/Cast?
(lstm_1/while/lstm_cell_1/dropout_1/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_1/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2*
(lstm_1/while/lstm_cell_1/dropout_1/Mul_1?
(lstm_1/while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2*
(lstm_1/while/lstm_cell_1/dropout_2/Const?
&lstm_1/while/lstm_cell_1/dropout_2/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2(
&lstm_1/while/lstm_cell_1/dropout_2/Mul?
(lstm_1/while/lstm_cell_1/dropout_2/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/dropout_2/Shape?
?lstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2A
?lstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
1lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>23
1lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/y?
/lstm_1/while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????21
/lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual?
'lstm_1/while/lstm_cell_1/dropout_2/CastCast3lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2)
'lstm_1/while/lstm_cell_1/dropout_2/Cast?
(lstm_1/while/lstm_cell_1/dropout_2/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_2/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2*
(lstm_1/while/lstm_cell_1/dropout_2/Mul_1?
(lstm_1/while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2*
(lstm_1/while/lstm_cell_1/dropout_3/Const?
&lstm_1/while/lstm_cell_1/dropout_3/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2(
&lstm_1/while/lstm_cell_1/dropout_3/Mul?
(lstm_1/while/lstm_cell_1/dropout_3/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/dropout_3/Shape?
?lstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2A
?lstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
1lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>23
1lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/y?
/lstm_1/while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????21
/lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual?
'lstm_1/while/lstm_cell_1/dropout_3/CastCast3lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2)
'lstm_1/while/lstm_cell_1/dropout_3/Cast?
(lstm_1/while/lstm_cell_1/dropout_3/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_3/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2*
(lstm_1/while/lstm_cell_1/dropout_3/Mul_1?
lstm_1/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_1/while/lstm_cell_1/Const?
(lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_1/while/lstm_cell_1/split/split_dim?
-lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02/
-lstm_1/while/lstm_cell_1/split/ReadVariableOp?
lstm_1/while/lstm_cell_1/splitSplit1lstm_1/while/lstm_cell_1/split/split_dim:output:05lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2 
lstm_1/while/lstm_cell_1/split?
lstm_1/while/lstm_cell_1/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2!
lstm_1/while/lstm_cell_1/MatMul?
!lstm_1/while/lstm_cell_1/MatMul_1MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_1?
!lstm_1/while/lstm_cell_1/MatMul_2MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_2?
!lstm_1/while/lstm_cell_1/MatMul_3MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_3?
 lstm_1/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_1/while/lstm_cell_1/Const_1?
*lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_1/while/lstm_cell_1/split_1/split_dim?
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype021
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
 lstm_1/while/lstm_cell_1/split_1Split3lstm_1/while/lstm_cell_1/split_1/split_dim:output:07lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2"
 lstm_1/while/lstm_cell_1/split_1?
 lstm_1/while/lstm_cell_1/BiasAddBiasAdd)lstm_1/while/lstm_cell_1/MatMul:product:0)lstm_1/while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/while/lstm_cell_1/BiasAdd?
"lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd+lstm_1/while/lstm_cell_1/MatMul_1:product:0)lstm_1/while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_1?
"lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd+lstm_1/while/lstm_cell_1/MatMul_2:product:0)lstm_1/while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_2?
"lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd+lstm_1/while/lstm_cell_1/MatMul_3:product:0)lstm_1/while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_3?
lstm_1/while/lstm_cell_1/mulMullstm_1_while_placeholder_3*lstm_1/while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/mul?
lstm_1/while/lstm_cell_1/mul_1Mullstm_1_while_placeholder_3,lstm_1/while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_1?
lstm_1/while/lstm_cell_1/mul_2Mullstm_1_while_placeholder_3,lstm_1/while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_2?
lstm_1/while/lstm_cell_1/mul_3Mullstm_1_while_placeholder_3,lstm_1/while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_3?
'lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02)
'lstm_1/while/lstm_cell_1/ReadVariableOp?
,lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_1/while/lstm_cell_1/strided_slice/stack?
.lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice/stack_1?
.lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell_1/strided_slice/stack_2?
&lstm_1/while/lstm_cell_1/strided_sliceStridedSlice/lstm_1/while/lstm_cell_1/ReadVariableOp:value:05lstm_1/while/lstm_cell_1/strided_slice/stack:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell_1/strided_slice?
!lstm_1/while/lstm_cell_1/MatMul_4MatMul lstm_1/while/lstm_cell_1/mul:z:0/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_4?
lstm_1/while/lstm_cell_1/addAddV2)lstm_1/while/lstm_cell_1/BiasAdd:output:0+lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/add?
 lstm_1/while/lstm_cell_1/SigmoidSigmoid lstm_1/while/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/while/lstm_cell_1/Sigmoid?
)lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_1?
.lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice_1/stack?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:07lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_1?
!lstm_1/while/lstm_cell_1/MatMul_5MatMul"lstm_1/while/lstm_cell_1/mul_1:z:01lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_5?
lstm_1/while/lstm_cell_1/add_1AddV2+lstm_1/while/lstm_cell_1/BiasAdd_1:output:0+lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_1?
"lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid"lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/Sigmoid_1?
lstm_1/while/lstm_cell_1/mul_4Mul&lstm_1/while/lstm_cell_1/Sigmoid_1:y:0lstm_1_while_placeholder_4*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_4?
)lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_2?
.lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   20
.lstm_1/while/lstm_cell_1/strided_slice_2/stack?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:07lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_2?
!lstm_1/while/lstm_cell_1/MatMul_6MatMul"lstm_1/while/lstm_cell_1/mul_2:z:01lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_6?
lstm_1/while/lstm_cell_1/add_2AddV2+lstm_1/while/lstm_cell_1/BiasAdd_2:output:0+lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_2?
lstm_1/while/lstm_cell_1/TanhTanh"lstm_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/Tanh?
lstm_1/while/lstm_cell_1/mul_5Mul$lstm_1/while/lstm_cell_1/Sigmoid:y:0!lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_5?
lstm_1/while/lstm_cell_1/add_3AddV2"lstm_1/while/lstm_cell_1/mul_4:z:0"lstm_1/while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_3?
)lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_3?
.lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   20
.lstm_1/while/lstm_cell_1/strided_slice_3/stack?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:07lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_3?
!lstm_1/while/lstm_cell_1/MatMul_7MatMul"lstm_1/while/lstm_cell_1/mul_3:z:01lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_7?
lstm_1/while/lstm_cell_1/add_4AddV2+lstm_1/while/lstm_cell_1/BiasAdd_3:output:0+lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_4?
"lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid"lstm_1/while/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/Sigmoid_2?
lstm_1/while/lstm_cell_1/Tanh_1Tanh"lstm_1/while/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2!
lstm_1/while/lstm_cell_1/Tanh_1?
lstm_1/while/lstm_cell_1/mul_6Mul&lstm_1/while/lstm_cell_1/Sigmoid_2:y:0#lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_6?
lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile/multiples?
lstm_1/while/TileTile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0$lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile?
lstm_1/while/SelectV2SelectV2lstm_1/while/Tile:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_2*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2?
lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_1/multiples?
lstm_1/while/Tile_1Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_1?
lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_2/multiples?
lstm_1/while/Tile_2Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_2?
lstm_1/while/SelectV2_1SelectV2lstm_1/while/Tile_1:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_3*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2_1?
lstm_1/while/SelectV2_2SelectV2lstm_1/while/Tile_2:output:0"lstm_1/while/lstm_cell_1/add_3:z:0lstm_1_while_placeholder_4*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2_2?
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholderlstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype023
1lstm_1/while/TensorArrayV2Write/TensorListSetItemj
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add/y?
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/addn
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add_1/y?
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/add_1?
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity?
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_1?
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_2?
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_3?
lstm_1/while/Identity_4Identitylstm_1/while/SelectV2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_4?
lstm_1/while/Identity_5Identity lstm_1/while/SelectV2_1:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_5?
lstm_1/while/Identity_6Identity lstm_1/while/SelectV2_2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_6"7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0";
lstm_1_while_identity_6 lstm_1/while/Identity_6:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"f
0lstm_1_while_lstm_cell_1_readvariableop_resource2lstm_1_while_lstm_cell_1_readvariableop_resource_0"v
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_1_split_readvariableop_resource8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"?
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensorelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :?????????:?????????:?????????: : : :::2R
'lstm_1/while/lstm_cell_1/ReadVariableOp'lstm_1/while/lstm_cell_1/ReadVariableOp2V
)lstm_1/while/lstm_cell_1/ReadVariableOp_1)lstm_1/while/lstm_cell_1/ReadVariableOp_12V
)lstm_1/while/lstm_cell_1/ReadVariableOp_2)lstm_1/while/lstm_cell_1/ReadVariableOp_22V
)lstm_1/while/lstm_cell_1/ReadVariableOp_3)lstm_1/while/lstm_cell_1/ReadVariableOp_32^
-lstm_1/while/lstm_cell_1/split/ReadVariableOp-lstm_1/while/lstm_cell_1/split/ReadVariableOp2b
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
??
?

while_body_14164143
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/ones_liket
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mulMulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_3?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_4*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_4?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_6}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2

while/Tile?
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/SelectV2?
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_1?
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_2?
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/SelectV2_1?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_1/add_3:z:0while_placeholder_4*
T0*'
_output_shapes
:?????????2
while/SelectV2_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/SelectV2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/SelectV2_1:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5?
while/Identity_6Identitywhile/SelectV2_2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_6")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :?????????:?????????:?????????: : : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?	
?
while_cond_14163239
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_16
2while_while_cond_14163239___redundant_placeholder06
2while_while_cond_14163239___redundant_placeholder16
2while_while_cond_14163239___redundant_placeholder26
2while_while_cond_14163239___redundant_placeholder36
2while_while_cond_14163239___redundant_placeholder4
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*j
_input_shapesY
W: : : : :?????????:?????????:?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?H
?
I__inference_lstm_cell_1_layer_call_and_return_conditional_losses_14164596

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpZ
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
	ones_likeP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:P*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:P*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????2
	BiasAdd_3a
mulMulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????2
mule
mul_1Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????2
mul_1e
mul_2Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????2
mul_2e
mul_3Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????2
mul_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mul_4|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh^
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_6?
IdentityIdentity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
lstm_1_while_cond_14161302*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4,
(lstm_1_while_less_lstm_1_strided_slice_1D
@lstm_1_while_lstm_1_while_cond_14161302___redundant_placeholder0D
@lstm_1_while_lstm_1_while_cond_14161302___redundant_placeholder1D
@lstm_1_while_lstm_1_while_cond_14161302___redundant_placeholder2D
@lstm_1_while_lstm_1_while_cond_14161302___redundant_placeholder3D
@lstm_1_while_lstm_1_while_cond_14161302___redundant_placeholder4
lstm_1_while_identity
?
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: 2
lstm_1/while/Lessr
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_1/while/Identity"7
lstm_1_while_identitylstm_1/while/Identity:output:0*j
_input_shapesY
W: : : : :?????????:?????????:?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?	
H
,__inference_masking_1_layer_call_fn_14161882

inputs
identity]

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

NotEqual/y|
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Any/reduction_indices?
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Anyp
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
Castb
mulMulinputsCast:y:0*
T0*4
_output_shapes"
 :??????????????????2
mul?
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:??????????????????*
squeeze_dims

?????????2	
Squeezeh
IdentityIdentitymul:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?0
?
8__inference_batch_normalization_1_layer_call_fn_14161974

inputs
assignmovingavg_14161949
assignmovingavg_1_14161955)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/14161949*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_14161949*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/14161949*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/14161949*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_14161949AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/14161949*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/14161955*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_14161955*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/14161955*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/14161955*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_14161955AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/14161955*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?l
?
I__inference_lstm_cell_1_layer_call_and_return_conditional_losses_14164519

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpZ
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ǩ?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ѳ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2Ԗ?2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_3/Mul_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:P*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:P*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????2
	BiasAdd_3`
mulMulstates_0dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
mulf
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
mul_1f
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
mul_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mul_4|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh^
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_6?
IdentityIdentity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
??
?
'sequential_1_lstm_1_while_body_14157173D
@sequential_1_lstm_1_while_sequential_1_lstm_1_while_loop_counterJ
Fsequential_1_lstm_1_while_sequential_1_lstm_1_while_maximum_iterations)
%sequential_1_lstm_1_while_placeholder+
'sequential_1_lstm_1_while_placeholder_1+
'sequential_1_lstm_1_while_placeholder_2+
'sequential_1_lstm_1_while_placeholder_3+
'sequential_1_lstm_1_while_placeholder_4C
?sequential_1_lstm_1_while_sequential_1_lstm_1_strided_slice_1_0
{sequential_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_tensorlistfromtensor_0?
sequential_1_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0I
Esequential_1_lstm_1_while_lstm_cell_1_split_readvariableop_resource_0K
Gsequential_1_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0C
?sequential_1_lstm_1_while_lstm_cell_1_readvariableop_resource_0&
"sequential_1_lstm_1_while_identity(
$sequential_1_lstm_1_while_identity_1(
$sequential_1_lstm_1_while_identity_2(
$sequential_1_lstm_1_while_identity_3(
$sequential_1_lstm_1_while_identity_4(
$sequential_1_lstm_1_while_identity_5(
$sequential_1_lstm_1_while_identity_6A
=sequential_1_lstm_1_while_sequential_1_lstm_1_strided_slice_1}
ysequential_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_tensorlistfromtensor?
}sequential_1_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_1_tensorlistfromtensorG
Csequential_1_lstm_1_while_lstm_cell_1_split_readvariableop_resourceI
Esequential_1_lstm_1_while_lstm_cell_1_split_1_readvariableop_resourceA
=sequential_1_lstm_1_while_lstm_cell_1_readvariableop_resource??4sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp?6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_1?6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_2?6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3?:sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp?<sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
Ksequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Ksequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_1_while_placeholderTsequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02?
=sequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem?
Msequential_1/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2O
Msequential_1/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
?sequential_1/lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemsequential_1_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0%sequential_1_lstm_1_while_placeholderVsequential_1/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
2A
?sequential_1/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem?
5sequential_1/lstm_1/while/lstm_cell_1/ones_like/ShapeShape'sequential_1_lstm_1_while_placeholder_3*
T0*
_output_shapes
:27
5sequential_1/lstm_1/while/lstm_cell_1/ones_like/Shape?
5sequential_1/lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??27
5sequential_1/lstm_1/while/lstm_cell_1/ones_like/Const?
/sequential_1/lstm_1/while/lstm_cell_1/ones_likeFill>sequential_1/lstm_1/while/lstm_cell_1/ones_like/Shape:output:0>sequential_1/lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????21
/sequential_1/lstm_1/while/lstm_cell_1/ones_like?
+sequential_1/lstm_1/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential_1/lstm_1/while/lstm_cell_1/Const?
5sequential_1/lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_1/lstm_1/while/lstm_cell_1/split/split_dim?
:sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOpEsequential_1_lstm_1_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02<
:sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp?
+sequential_1/lstm_1/while/lstm_cell_1/splitSplit>sequential_1/lstm_1/while/lstm_cell_1/split/split_dim:output:0Bsequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2-
+sequential_1/lstm_1/while/lstm_cell_1/split?
,sequential_1/lstm_1/while/lstm_cell_1/MatMulMatMulDsequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2.
,sequential_1/lstm_1/while/lstm_cell_1/MatMul?
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_1MatMulDsequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????20
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_1?
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_2MatMulDsequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????20
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_2?
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_3MatMulDsequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????20
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_3?
-sequential_1/lstm_1/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_1/lstm_1/while/lstm_cell_1/Const_1?
7sequential_1/lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_1/lstm_1/while/lstm_cell_1/split_1/split_dim?
<sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOpGsequential_1_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype02>
<sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
-sequential_1/lstm_1/while/lstm_cell_1/split_1Split@sequential_1/lstm_1/while/lstm_cell_1/split_1/split_dim:output:0Dsequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2/
-sequential_1/lstm_1/while/lstm_cell_1/split_1?
-sequential_1/lstm_1/while/lstm_cell_1/BiasAddBiasAdd6sequential_1/lstm_1/while/lstm_cell_1/MatMul:product:06sequential_1/lstm_1/while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2/
-sequential_1/lstm_1/while/lstm_cell_1/BiasAdd?
/sequential_1/lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd8sequential_1/lstm_1/while/lstm_cell_1/MatMul_1:product:06sequential_1/lstm_1/while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????21
/sequential_1/lstm_1/while/lstm_cell_1/BiasAdd_1?
/sequential_1/lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd8sequential_1/lstm_1/while/lstm_cell_1/MatMul_2:product:06sequential_1/lstm_1/while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????21
/sequential_1/lstm_1/while/lstm_cell_1/BiasAdd_2?
/sequential_1/lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd8sequential_1/lstm_1/while/lstm_cell_1/MatMul_3:product:06sequential_1/lstm_1/while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????21
/sequential_1/lstm_1/while/lstm_cell_1/BiasAdd_3?
)sequential_1/lstm_1/while/lstm_cell_1/mulMul'sequential_1_lstm_1_while_placeholder_38sequential_1/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2+
)sequential_1/lstm_1/while/lstm_cell_1/mul?
+sequential_1/lstm_1/while/lstm_cell_1/mul_1Mul'sequential_1_lstm_1_while_placeholder_38sequential_1/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/lstm_1/while/lstm_cell_1/mul_1?
+sequential_1/lstm_1/while/lstm_cell_1/mul_2Mul'sequential_1_lstm_1_while_placeholder_38sequential_1/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/lstm_1/while/lstm_cell_1/mul_2?
+sequential_1/lstm_1/while/lstm_cell_1/mul_3Mul'sequential_1_lstm_1_while_placeholder_38sequential_1/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/lstm_1/while/lstm_cell_1/mul_3?
4sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp?sequential_1_lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype026
4sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp?
9sequential_1/lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2;
9sequential_1/lstm_1/while/lstm_cell_1/strided_slice/stack?
;sequential_1/lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2=
;sequential_1/lstm_1/while/lstm_cell_1/strided_slice/stack_1?
;sequential_1/lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;sequential_1/lstm_1/while/lstm_cell_1/strided_slice/stack_2?
3sequential_1/lstm_1/while/lstm_cell_1/strided_sliceStridedSlice<sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp:value:0Bsequential_1/lstm_1/while/lstm_cell_1/strided_slice/stack:output:0Dsequential_1/lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:0Dsequential_1/lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask25
3sequential_1/lstm_1/while/lstm_cell_1/strided_slice?
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_4MatMul-sequential_1/lstm_1/while/lstm_cell_1/mul:z:0<sequential_1/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????20
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_4?
)sequential_1/lstm_1/while/lstm_cell_1/addAddV26sequential_1/lstm_1/while/lstm_cell_1/BiasAdd:output:08sequential_1/lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2+
)sequential_1/lstm_1/while/lstm_cell_1/add?
-sequential_1/lstm_1/while/lstm_cell_1/SigmoidSigmoid-sequential_1/lstm_1/while/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2/
-sequential_1/lstm_1/while/lstm_cell_1/Sigmoid?
6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp?sequential_1_lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype028
6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_1?
;sequential_1/lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2=
;sequential_1/lstm_1/while/lstm_cell_1/strided_slice_1/stack?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_1/stack_1?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_1/stack_2?
5sequential_1/lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice>sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:0Dsequential_1/lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:0Fsequential_1/lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:0Fsequential_1/lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask27
5sequential_1/lstm_1/while/lstm_cell_1/strided_slice_1?
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_5MatMul/sequential_1/lstm_1/while/lstm_cell_1/mul_1:z:0>sequential_1/lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????20
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_5?
+sequential_1/lstm_1/while/lstm_cell_1/add_1AddV28sequential_1/lstm_1/while/lstm_cell_1/BiasAdd_1:output:08sequential_1/lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/lstm_1/while/lstm_cell_1/add_1?
/sequential_1/lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid/sequential_1/lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????21
/sequential_1/lstm_1/while/lstm_cell_1/Sigmoid_1?
+sequential_1/lstm_1/while/lstm_cell_1/mul_4Mul3sequential_1/lstm_1/while/lstm_cell_1/Sigmoid_1:y:0'sequential_1_lstm_1_while_placeholder_4*
T0*'
_output_shapes
:?????????2-
+sequential_1/lstm_1/while/lstm_cell_1/mul_4?
6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp?sequential_1_lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype028
6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_2?
;sequential_1/lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2=
;sequential_1/lstm_1/while/lstm_cell_1/strided_slice_2/stack?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_2/stack_1?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_2/stack_2?
5sequential_1/lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice>sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:0Dsequential_1/lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:0Fsequential_1/lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:0Fsequential_1/lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask27
5sequential_1/lstm_1/while/lstm_cell_1/strided_slice_2?
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_6MatMul/sequential_1/lstm_1/while/lstm_cell_1/mul_2:z:0>sequential_1/lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????20
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_6?
+sequential_1/lstm_1/while/lstm_cell_1/add_2AddV28sequential_1/lstm_1/while/lstm_cell_1/BiasAdd_2:output:08sequential_1/lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/lstm_1/while/lstm_cell_1/add_2?
*sequential_1/lstm_1/while/lstm_cell_1/TanhTanh/sequential_1/lstm_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2,
*sequential_1/lstm_1/while/lstm_cell_1/Tanh?
+sequential_1/lstm_1/while/lstm_cell_1/mul_5Mul1sequential_1/lstm_1/while/lstm_cell_1/Sigmoid:y:0.sequential_1/lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/lstm_1/while/lstm_cell_1/mul_5?
+sequential_1/lstm_1/while/lstm_cell_1/add_3AddV2/sequential_1/lstm_1/while/lstm_cell_1/mul_4:z:0/sequential_1/lstm_1/while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/lstm_1/while/lstm_cell_1/add_3?
6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp?sequential_1_lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype028
6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3?
;sequential_1/lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2=
;sequential_1/lstm_1/while/lstm_cell_1/strided_slice_3/stack?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_3/stack_1?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_1/lstm_1/while/lstm_cell_1/strided_slice_3/stack_2?
5sequential_1/lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice>sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:0Dsequential_1/lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:0Fsequential_1/lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:0Fsequential_1/lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask27
5sequential_1/lstm_1/while/lstm_cell_1/strided_slice_3?
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_7MatMul/sequential_1/lstm_1/while/lstm_cell_1/mul_3:z:0>sequential_1/lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????20
.sequential_1/lstm_1/while/lstm_cell_1/MatMul_7?
+sequential_1/lstm_1/while/lstm_cell_1/add_4AddV28sequential_1/lstm_1/while/lstm_cell_1/BiasAdd_3:output:08sequential_1/lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/lstm_1/while/lstm_cell_1/add_4?
/sequential_1/lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid/sequential_1/lstm_1/while/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????21
/sequential_1/lstm_1/while/lstm_cell_1/Sigmoid_2?
,sequential_1/lstm_1/while/lstm_cell_1/Tanh_1Tanh/sequential_1/lstm_1/while/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2.
,sequential_1/lstm_1/while/lstm_cell_1/Tanh_1?
+sequential_1/lstm_1/while/lstm_cell_1/mul_6Mul3sequential_1/lstm_1/while/lstm_cell_1/Sigmoid_2:y:00sequential_1/lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/lstm_1/while/lstm_cell_1/mul_6?
(sequential_1/lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2*
(sequential_1/lstm_1/while/Tile/multiples?
sequential_1/lstm_1/while/TileTileFsequential_1/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:01sequential_1/lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2 
sequential_1/lstm_1/while/Tile?
"sequential_1/lstm_1/while/SelectV2SelectV2'sequential_1/lstm_1/while/Tile:output:0/sequential_1/lstm_1/while/lstm_cell_1/mul_6:z:0'sequential_1_lstm_1_while_placeholder_2*
T0*'
_output_shapes
:?????????2$
"sequential_1/lstm_1/while/SelectV2?
*sequential_1/lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*sequential_1/lstm_1/while/Tile_1/multiples?
 sequential_1/lstm_1/while/Tile_1TileFsequential_1/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:03sequential_1/lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2"
 sequential_1/lstm_1/while/Tile_1?
*sequential_1/lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*sequential_1/lstm_1/while/Tile_2/multiples?
 sequential_1/lstm_1/while/Tile_2TileFsequential_1/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:03sequential_1/lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2"
 sequential_1/lstm_1/while/Tile_2?
$sequential_1/lstm_1/while/SelectV2_1SelectV2)sequential_1/lstm_1/while/Tile_1:output:0/sequential_1/lstm_1/while/lstm_cell_1/mul_6:z:0'sequential_1_lstm_1_while_placeholder_3*
T0*'
_output_shapes
:?????????2&
$sequential_1/lstm_1/while/SelectV2_1?
$sequential_1/lstm_1/while/SelectV2_2SelectV2)sequential_1/lstm_1/while/Tile_2:output:0/sequential_1/lstm_1/while/lstm_cell_1/add_3:z:0'sequential_1_lstm_1_while_placeholder_4*
T0*'
_output_shapes
:?????????2&
$sequential_1/lstm_1/while/SelectV2_2?
>sequential_1/lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_1_while_placeholder_1%sequential_1_lstm_1_while_placeholder+sequential_1/lstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype02@
>sequential_1/lstm_1/while/TensorArrayV2Write/TensorListSetItem?
sequential_1/lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_1/lstm_1/while/add/y?
sequential_1/lstm_1/while/addAddV2%sequential_1_lstm_1_while_placeholder(sequential_1/lstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_1/while/add?
!sequential_1/lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_1/lstm_1/while/add_1/y?
sequential_1/lstm_1/while/add_1AddV2@sequential_1_lstm_1_while_sequential_1_lstm_1_while_loop_counter*sequential_1/lstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_1/while/add_1?
"sequential_1/lstm_1/while/IdentityIdentity#sequential_1/lstm_1/while/add_1:z:05^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp7^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_17^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_27^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3;^sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp=^sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2$
"sequential_1/lstm_1/while/Identity?
$sequential_1/lstm_1/while/Identity_1IdentityFsequential_1_lstm_1_while_sequential_1_lstm_1_while_maximum_iterations5^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp7^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_17^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_27^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3;^sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp=^sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_1/while/Identity_1?
$sequential_1/lstm_1/while/Identity_2Identity!sequential_1/lstm_1/while/add:z:05^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp7^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_17^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_27^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3;^sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp=^sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_1/while/Identity_2?
$sequential_1/lstm_1/while/Identity_3IdentityNsequential_1/lstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:05^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp7^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_17^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_27^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3;^sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp=^sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_1/while/Identity_3?
$sequential_1/lstm_1/while/Identity_4Identity+sequential_1/lstm_1/while/SelectV2:output:05^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp7^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_17^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_27^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3;^sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp=^sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2&
$sequential_1/lstm_1/while/Identity_4?
$sequential_1/lstm_1/while/Identity_5Identity-sequential_1/lstm_1/while/SelectV2_1:output:05^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp7^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_17^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_27^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3;^sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp=^sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2&
$sequential_1/lstm_1/while/Identity_5?
$sequential_1/lstm_1/while/Identity_6Identity-sequential_1/lstm_1/while/SelectV2_2:output:05^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp7^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_17^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_27^sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_3;^sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp=^sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2&
$sequential_1/lstm_1/while/Identity_6"Q
"sequential_1_lstm_1_while_identity+sequential_1/lstm_1/while/Identity:output:0"U
$sequential_1_lstm_1_while_identity_1-sequential_1/lstm_1/while/Identity_1:output:0"U
$sequential_1_lstm_1_while_identity_2-sequential_1/lstm_1/while/Identity_2:output:0"U
$sequential_1_lstm_1_while_identity_3-sequential_1/lstm_1/while/Identity_3:output:0"U
$sequential_1_lstm_1_while_identity_4-sequential_1/lstm_1/while/Identity_4:output:0"U
$sequential_1_lstm_1_while_identity_5-sequential_1/lstm_1/while/Identity_5:output:0"U
$sequential_1_lstm_1_while_identity_6-sequential_1/lstm_1/while/Identity_6:output:0"?
=sequential_1_lstm_1_while_lstm_cell_1_readvariableop_resource?sequential_1_lstm_1_while_lstm_cell_1_readvariableop_resource_0"?
Esequential_1_lstm_1_while_lstm_cell_1_split_1_readvariableop_resourceGsequential_1_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"?
Csequential_1_lstm_1_while_lstm_cell_1_split_readvariableop_resourceEsequential_1_lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"?
=sequential_1_lstm_1_while_sequential_1_lstm_1_strided_slice_1?sequential_1_lstm_1_while_sequential_1_lstm_1_strided_slice_1_0"?
}sequential_1_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_1_tensorlistfromtensorsequential_1_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
ysequential_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :?????????:?????????:?????????: : : :::2l
4sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp4sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp2p
6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_16sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_12p
6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_26sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_22p
6sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_36sequential_1/lstm_1/while/lstm_cell_1/ReadVariableOp_32x
:sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp:sequential_1/lstm_1/while/lstm_cell_1/split/ReadVariableOp2|
<sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp<sequential_1/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
?
lstm_1_while_cond_14160935*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4,
(lstm_1_while_less_lstm_1_strided_slice_1D
@lstm_1_while_lstm_1_while_cond_14160935___redundant_placeholder0D
@lstm_1_while_lstm_1_while_cond_14160935___redundant_placeholder1D
@lstm_1_while_lstm_1_while_cond_14160935___redundant_placeholder2D
@lstm_1_while_lstm_1_while_cond_14160935___redundant_placeholder3D
@lstm_1_while_lstm_1_while_cond_14160935___redundant_placeholder4
lstm_1_while_identity
?
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: 2
lstm_1/while/Lessr
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_1/while/Identity"7
lstm_1_while_identitylstm_1/while/Identity:output:0*j
_input_shapesY
W: : : : :?????????:?????????:?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
??
?
lstm_1_while_body_14161653*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0i
elstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0<
8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0>
:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_06
2lstm_1_while_lstm_cell_1_readvariableop_resource_0
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5
lstm_1_while_identity_6'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorg
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor:
6lstm_1_while_lstm_cell_1_split_readvariableop_resource<
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource4
0lstm_1_while_lstm_cell_1_readvariableop_resource??'lstm_1/while/lstm_cell_1/ReadVariableOp?)lstm_1/while/lstm_cell_1/ReadVariableOp_1?)lstm_1/while/lstm_cell_1/ReadVariableOp_2?)lstm_1/while/lstm_cell_1/ReadVariableOp_3?-lstm_1/while/lstm_cell_1/split/ReadVariableOp?/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0lstm_1/while/TensorArrayV2Read/TensorListGetItem?
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0lstm_1_while_placeholderIlstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
24
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItem?
(lstm_1/while/lstm_cell_1/ones_like/ShapeShapelstm_1_while_placeholder_3*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/ones_like/Shape?
(lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(lstm_1/while/lstm_cell_1/ones_like/Const?
"lstm_1/while/lstm_cell_1/ones_likeFill1lstm_1/while/lstm_cell_1/ones_like/Shape:output:01lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/ones_like?
lstm_1/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_1/while/lstm_cell_1/Const?
(lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_1/while/lstm_cell_1/split/split_dim?
-lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02/
-lstm_1/while/lstm_cell_1/split/ReadVariableOp?
lstm_1/while/lstm_cell_1/splitSplit1lstm_1/while/lstm_cell_1/split/split_dim:output:05lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2 
lstm_1/while/lstm_cell_1/split?
lstm_1/while/lstm_cell_1/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2!
lstm_1/while/lstm_cell_1/MatMul?
!lstm_1/while/lstm_cell_1/MatMul_1MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_1?
!lstm_1/while/lstm_cell_1/MatMul_2MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_2?
!lstm_1/while/lstm_cell_1/MatMul_3MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_3?
 lstm_1/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_1/while/lstm_cell_1/Const_1?
*lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_1/while/lstm_cell_1/split_1/split_dim?
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype021
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
 lstm_1/while/lstm_cell_1/split_1Split3lstm_1/while/lstm_cell_1/split_1/split_dim:output:07lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2"
 lstm_1/while/lstm_cell_1/split_1?
 lstm_1/while/lstm_cell_1/BiasAddBiasAdd)lstm_1/while/lstm_cell_1/MatMul:product:0)lstm_1/while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/while/lstm_cell_1/BiasAdd?
"lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd+lstm_1/while/lstm_cell_1/MatMul_1:product:0)lstm_1/while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_1?
"lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd+lstm_1/while/lstm_cell_1/MatMul_2:product:0)lstm_1/while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_2?
"lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd+lstm_1/while/lstm_cell_1/MatMul_3:product:0)lstm_1/while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_3?
lstm_1/while/lstm_cell_1/mulMullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/mul?
lstm_1/while/lstm_cell_1/mul_1Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_1?
lstm_1/while/lstm_cell_1/mul_2Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_2?
lstm_1/while/lstm_cell_1/mul_3Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_3?
'lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02)
'lstm_1/while/lstm_cell_1/ReadVariableOp?
,lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_1/while/lstm_cell_1/strided_slice/stack?
.lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice/stack_1?
.lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell_1/strided_slice/stack_2?
&lstm_1/while/lstm_cell_1/strided_sliceStridedSlice/lstm_1/while/lstm_cell_1/ReadVariableOp:value:05lstm_1/while/lstm_cell_1/strided_slice/stack:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell_1/strided_slice?
!lstm_1/while/lstm_cell_1/MatMul_4MatMul lstm_1/while/lstm_cell_1/mul:z:0/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_4?
lstm_1/while/lstm_cell_1/addAddV2)lstm_1/while/lstm_cell_1/BiasAdd:output:0+lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/add?
 lstm_1/while/lstm_cell_1/SigmoidSigmoid lstm_1/while/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/while/lstm_cell_1/Sigmoid?
)lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_1?
.lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice_1/stack?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:07lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_1?
!lstm_1/while/lstm_cell_1/MatMul_5MatMul"lstm_1/while/lstm_cell_1/mul_1:z:01lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_5?
lstm_1/while/lstm_cell_1/add_1AddV2+lstm_1/while/lstm_cell_1/BiasAdd_1:output:0+lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_1?
"lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid"lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/Sigmoid_1?
lstm_1/while/lstm_cell_1/mul_4Mul&lstm_1/while/lstm_cell_1/Sigmoid_1:y:0lstm_1_while_placeholder_4*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_4?
)lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_2?
.lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   20
.lstm_1/while/lstm_cell_1/strided_slice_2/stack?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:07lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_2?
!lstm_1/while/lstm_cell_1/MatMul_6MatMul"lstm_1/while/lstm_cell_1/mul_2:z:01lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_6?
lstm_1/while/lstm_cell_1/add_2AddV2+lstm_1/while/lstm_cell_1/BiasAdd_2:output:0+lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_2?
lstm_1/while/lstm_cell_1/TanhTanh"lstm_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/Tanh?
lstm_1/while/lstm_cell_1/mul_5Mul$lstm_1/while/lstm_cell_1/Sigmoid:y:0!lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_5?
lstm_1/while/lstm_cell_1/add_3AddV2"lstm_1/while/lstm_cell_1/mul_4:z:0"lstm_1/while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_3?
)lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_3?
.lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   20
.lstm_1/while/lstm_cell_1/strided_slice_3/stack?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:07lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_3?
!lstm_1/while/lstm_cell_1/MatMul_7MatMul"lstm_1/while/lstm_cell_1/mul_3:z:01lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_7?
lstm_1/while/lstm_cell_1/add_4AddV2+lstm_1/while/lstm_cell_1/BiasAdd_3:output:0+lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_4?
"lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid"lstm_1/while/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/Sigmoid_2?
lstm_1/while/lstm_cell_1/Tanh_1Tanh"lstm_1/while/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2!
lstm_1/while/lstm_cell_1/Tanh_1?
lstm_1/while/lstm_cell_1/mul_6Mul&lstm_1/while/lstm_cell_1/Sigmoid_2:y:0#lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_6?
lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile/multiples?
lstm_1/while/TileTile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0$lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile?
lstm_1/while/SelectV2SelectV2lstm_1/while/Tile:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_2*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2?
lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_1/multiples?
lstm_1/while/Tile_1Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_1?
lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_2/multiples?
lstm_1/while/Tile_2Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_2?
lstm_1/while/SelectV2_1SelectV2lstm_1/while/Tile_1:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_3*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2_1?
lstm_1/while/SelectV2_2SelectV2lstm_1/while/Tile_2:output:0"lstm_1/while/lstm_cell_1/add_3:z:0lstm_1_while_placeholder_4*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2_2?
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholderlstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype023
1lstm_1/while/TensorArrayV2Write/TensorListSetItemj
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add/y?
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/addn
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add_1/y?
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/add_1?
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity?
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_1?
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_2?
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_3?
lstm_1/while/Identity_4Identitylstm_1/while/SelectV2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_4?
lstm_1/while/Identity_5Identity lstm_1/while/SelectV2_1:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_5?
lstm_1/while/Identity_6Identity lstm_1/while/SelectV2_2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_6"7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0";
lstm_1_while_identity_6 lstm_1/while/Identity_6:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"f
0lstm_1_while_lstm_cell_1_readvariableop_resource2lstm_1_while_lstm_cell_1_readvariableop_resource_0"v
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_1_split_readvariableop_resource8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"?
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensorelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :?????????:?????????:?????????: : : :::2R
'lstm_1/while/lstm_cell_1/ReadVariableOp'lstm_1/while/lstm_cell_1/ReadVariableOp2V
)lstm_1/while/lstm_cell_1/ReadVariableOp_1)lstm_1/while/lstm_cell_1/ReadVariableOp_12V
)lstm_1/while/lstm_cell_1/ReadVariableOp_2)lstm_1/while/lstm_cell_1/ReadVariableOp_22V
)lstm_1/while/lstm_cell_1/ReadVariableOp_3)lstm_1/while/lstm_cell_1/ReadVariableOp_32^
-lstm_1/while/lstm_cell_1/split/ReadVariableOp-lstm_1/while/lstm_cell_1/split/ReadVariableOp2b
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
??
?

while_body_14163240
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/ones_like?
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2!
while/lstm_cell_1/dropout/Const?
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/dropout/Mul?
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_1/dropout/Shape?
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ڤ?28
6while/lstm_cell_1/dropout/random_uniform/RandomUniform?
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2*
(while/lstm_cell_1/dropout/GreaterEqual/y?
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2(
&while/lstm_cell_1/dropout/GreaterEqual?
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2 
while/lstm_cell_1/dropout/Cast?
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout/Mul_1?
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_1/Const?
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_1/Mul?
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_1/Shape?
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2̹?2:
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_1/GreaterEqual/y?
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_1/GreaterEqual?
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_1/Cast?
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_1/Mul_1?
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_2/Const?
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_2/Mul?
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_2/Shape?
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??p2:
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_2/GreaterEqual/y?
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_2/GreaterEqual?
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_2/Cast?
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_2/Mul_1?
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_3/Const?
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_3/Mul?
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_3/Shape?
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2?Ԝ2:
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_3/GreaterEqual/y?
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_3/GreaterEqual?
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_3/Cast?
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_3/Mul_1t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mulMulwhile_placeholder_3#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mulwhile_placeholder_3%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mulwhile_placeholder_3%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mulwhile_placeholder_3%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_3?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_4*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_4?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_6}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2

while/Tile?
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/SelectV2?
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_1?
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_2?
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/SelectV2_1?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_1/add_3:z:0while_placeholder_4*
T0*'
_output_shapes
:?????????2
while/SelectV2_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/SelectV2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/SelectV2_1:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5?
while/Identity_6Identitywhile/SelectV2_2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_6")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :?????????:?????????:?????????: : : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
??
?

while_body_14163541
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/ones_liket
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mulMulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_3?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_4*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_4?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_6}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2

while/Tile?
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????2
while/SelectV2?
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_1?
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
while/Tile_2?
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/SelectV2_1?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_1/add_3:z:0while_placeholder_4*
T0*'
_output_shapes
:?????????2
while/SelectV2_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/SelectV2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/SelectV2_1:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5?
while/Identity_6Identitywhile/SelectV2_2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_6")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :?????????:?????????:?????????: : : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?H
?
.__inference_lstm_cell_1_layer_call_fn_14164782

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpZ
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
	ones_likeP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:P*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:P*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????2
	BiasAdd_3a
mulMulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????2
mule
mul_1Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????2
mul_1e
mul_2Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????2
mul_2e
mul_3Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:?????????2
mul_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mul_4|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh^
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_6?
IdentityIdentity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14161938

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
*__inference_dense_1_layer_call_fn_14164310

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_14161860
masking_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmasking_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_141573472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:??????????????????:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
4
_output_shapes"
 :??????????????????
)
_user_specified_namemasking_1_input
??
?
lstm_1_while_body_14160936*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0i
elstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0<
8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0>
:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_06
2lstm_1_while_lstm_cell_1_readvariableop_resource_0
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5
lstm_1_while_identity_6'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorg
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor:
6lstm_1_while_lstm_cell_1_split_readvariableop_resource<
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource4
0lstm_1_while_lstm_cell_1_readvariableop_resource??'lstm_1/while/lstm_cell_1/ReadVariableOp?)lstm_1/while/lstm_cell_1/ReadVariableOp_1?)lstm_1/while/lstm_cell_1/ReadVariableOp_2?)lstm_1/while/lstm_cell_1/ReadVariableOp_3?-lstm_1/while/lstm_cell_1/split/ReadVariableOp?/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0lstm_1/while/TensorArrayV2Read/TensorListGetItem?
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0lstm_1_while_placeholderIlstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
24
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItem?
(lstm_1/while/lstm_cell_1/ones_like/ShapeShapelstm_1_while_placeholder_3*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/ones_like/Shape?
(lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(lstm_1/while/lstm_cell_1/ones_like/Const?
"lstm_1/while/lstm_cell_1/ones_likeFill1lstm_1/while/lstm_cell_1/ones_like/Shape:output:01lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/ones_like?
lstm_1/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_1/while/lstm_cell_1/Const?
(lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_1/while/lstm_cell_1/split/split_dim?
-lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02/
-lstm_1/while/lstm_cell_1/split/ReadVariableOp?
lstm_1/while/lstm_cell_1/splitSplit1lstm_1/while/lstm_cell_1/split/split_dim:output:05lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2 
lstm_1/while/lstm_cell_1/split?
lstm_1/while/lstm_cell_1/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2!
lstm_1/while/lstm_cell_1/MatMul?
!lstm_1/while/lstm_cell_1/MatMul_1MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_1?
!lstm_1/while/lstm_cell_1/MatMul_2MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_2?
!lstm_1/while/lstm_cell_1/MatMul_3MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_3?
 lstm_1/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_1/while/lstm_cell_1/Const_1?
*lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_1/while/lstm_cell_1/split_1/split_dim?
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype021
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
 lstm_1/while/lstm_cell_1/split_1Split3lstm_1/while/lstm_cell_1/split_1/split_dim:output:07lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2"
 lstm_1/while/lstm_cell_1/split_1?
 lstm_1/while/lstm_cell_1/BiasAddBiasAdd)lstm_1/while/lstm_cell_1/MatMul:product:0)lstm_1/while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/while/lstm_cell_1/BiasAdd?
"lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd+lstm_1/while/lstm_cell_1/MatMul_1:product:0)lstm_1/while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_1?
"lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd+lstm_1/while/lstm_cell_1/MatMul_2:product:0)lstm_1/while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_2?
"lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd+lstm_1/while/lstm_cell_1/MatMul_3:product:0)lstm_1/while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_3?
lstm_1/while/lstm_cell_1/mulMullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/mul?
lstm_1/while/lstm_cell_1/mul_1Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_1?
lstm_1/while/lstm_cell_1/mul_2Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_2?
lstm_1/while/lstm_cell_1/mul_3Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_3?
'lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02)
'lstm_1/while/lstm_cell_1/ReadVariableOp?
,lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_1/while/lstm_cell_1/strided_slice/stack?
.lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice/stack_1?
.lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell_1/strided_slice/stack_2?
&lstm_1/while/lstm_cell_1/strided_sliceStridedSlice/lstm_1/while/lstm_cell_1/ReadVariableOp:value:05lstm_1/while/lstm_cell_1/strided_slice/stack:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell_1/strided_slice?
!lstm_1/while/lstm_cell_1/MatMul_4MatMul lstm_1/while/lstm_cell_1/mul:z:0/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_4?
lstm_1/while/lstm_cell_1/addAddV2)lstm_1/while/lstm_cell_1/BiasAdd:output:0+lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/add?
 lstm_1/while/lstm_cell_1/SigmoidSigmoid lstm_1/while/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/while/lstm_cell_1/Sigmoid?
)lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_1?
.lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice_1/stack?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:07lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_1?
!lstm_1/while/lstm_cell_1/MatMul_5MatMul"lstm_1/while/lstm_cell_1/mul_1:z:01lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_5?
lstm_1/while/lstm_cell_1/add_1AddV2+lstm_1/while/lstm_cell_1/BiasAdd_1:output:0+lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_1?
"lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid"lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/Sigmoid_1?
lstm_1/while/lstm_cell_1/mul_4Mul&lstm_1/while/lstm_cell_1/Sigmoid_1:y:0lstm_1_while_placeholder_4*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_4?
)lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_2?
.lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   20
.lstm_1/while/lstm_cell_1/strided_slice_2/stack?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:07lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_2?
!lstm_1/while/lstm_cell_1/MatMul_6MatMul"lstm_1/while/lstm_cell_1/mul_2:z:01lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_6?
lstm_1/while/lstm_cell_1/add_2AddV2+lstm_1/while/lstm_cell_1/BiasAdd_2:output:0+lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_2?
lstm_1/while/lstm_cell_1/TanhTanh"lstm_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/Tanh?
lstm_1/while/lstm_cell_1/mul_5Mul$lstm_1/while/lstm_cell_1/Sigmoid:y:0!lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_5?
lstm_1/while/lstm_cell_1/add_3AddV2"lstm_1/while/lstm_cell_1/mul_4:z:0"lstm_1/while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_3?
)lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_3?
.lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   20
.lstm_1/while/lstm_cell_1/strided_slice_3/stack?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:07lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_3?
!lstm_1/while/lstm_cell_1/MatMul_7MatMul"lstm_1/while/lstm_cell_1/mul_3:z:01lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_7?
lstm_1/while/lstm_cell_1/add_4AddV2+lstm_1/while/lstm_cell_1/BiasAdd_3:output:0+lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_4?
"lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid"lstm_1/while/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/Sigmoid_2?
lstm_1/while/lstm_cell_1/Tanh_1Tanh"lstm_1/while/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2!
lstm_1/while/lstm_cell_1/Tanh_1?
lstm_1/while/lstm_cell_1/mul_6Mul&lstm_1/while/lstm_cell_1/Sigmoid_2:y:0#lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_6?
lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile/multiples?
lstm_1/while/TileTile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0$lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile?
lstm_1/while/SelectV2SelectV2lstm_1/while/Tile:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_2*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2?
lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_1/multiples?
lstm_1/while/Tile_1Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_1?
lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_2/multiples?
lstm_1/while/Tile_2Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_2?
lstm_1/while/SelectV2_1SelectV2lstm_1/while/Tile_1:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_3*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2_1?
lstm_1/while/SelectV2_2SelectV2lstm_1/while/Tile_2:output:0"lstm_1/while/lstm_cell_1/add_3:z:0lstm_1_while_placeholder_4*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2_2?
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholderlstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype023
1lstm_1/while/TensorArrayV2Write/TensorListSetItemj
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add/y?
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/addn
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add_1/y?
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/add_1?
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity?
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_1?
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_2?
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_3?
lstm_1/while/Identity_4Identitylstm_1/while/SelectV2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_4?
lstm_1/while/Identity_5Identity lstm_1/while/SelectV2_1:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_5?
lstm_1/while/Identity_6Identity lstm_1/while/SelectV2_2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_6"7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0";
lstm_1_while_identity_6 lstm_1/while/Identity_6:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"f
0lstm_1_while_lstm_cell_1_readvariableop_resource2lstm_1_while_lstm_cell_1_readvariableop_resource_0"v
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_1_split_readvariableop_resource8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"?
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensorelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :?????????:?????????:?????????: : : :::2R
'lstm_1/while/lstm_cell_1/ReadVariableOp'lstm_1/while/lstm_cell_1/ReadVariableOp2V
)lstm_1/while/lstm_cell_1/ReadVariableOp_1)lstm_1/while/lstm_cell_1/ReadVariableOp_12V
)lstm_1/while/lstm_cell_1/ReadVariableOp_2)lstm_1/while/lstm_cell_1/ReadVariableOp_22V
)lstm_1/while/lstm_cell_1/ReadVariableOp_3)lstm_1/while/lstm_cell_1/ReadVariableOp_32^
-lstm_1/while/lstm_cell_1/split/ReadVariableOp-lstm_1/while/lstm_cell_1/split/ReadVariableOp2b
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?	
?
while_cond_14163540
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_16
2while_while_cond_14163540___redundant_placeholder06
2while_while_cond_14163540___redundant_placeholder16
2while_while_cond_14163540___redundant_placeholder26
2while_while_cond_14163540___redundant_placeholder36
2while_while_cond_14163540___redundant_placeholder4
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*j
_input_shapesY
W: : : : :?????????:?????????:?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?
?
lstm_1_while_cond_14161652*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4,
(lstm_1_while_less_lstm_1_strided_slice_1D
@lstm_1_while_lstm_1_while_cond_14161652___redundant_placeholder0D
@lstm_1_while_lstm_1_while_cond_14161652___redundant_placeholder1D
@lstm_1_while_lstm_1_while_cond_14161652___redundant_placeholder2D
@lstm_1_while_lstm_1_while_cond_14161652___redundant_placeholder3D
@lstm_1_while_lstm_1_while_cond_14161652___redundant_placeholder4
lstm_1_while_identity
?
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: 2
lstm_1/while/Lessr
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_1/while/Identity"7
lstm_1_while_identitylstm_1/while/Identity:output:0*j
_input_shapesY
W: : : : :?????????:?????????:?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
??
?
)__inference_lstm_1_layer_call_fn_14164021

inputs
mask
-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????2
transpose_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
lstm_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout/Const?
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Mul?
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shape?
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??422
0lstm_cell_1/dropout/random_uniform/RandomUniform?
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2$
"lstm_cell_1/dropout/GreaterEqual/y?
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_cell_1/dropout/GreaterEqual?
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Cast?
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_1/Const?
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Mul?
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shape?
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??>24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform?
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_1/GreaterEqual/y?
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_1/GreaterEqual?
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Cast?
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_2/Const?
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Mul?
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shape?
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform?
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_2/GreaterEqual/y?
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_2/GreaterEqual?
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Cast?
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_3/Const?
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Mul?
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shape?
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ߌ?24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform?
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_3/GreaterEqual/y?
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_3/GreaterEqual?
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Cast?
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Mul_1h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mulMulzeros:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulzeros:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulzeros:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulzeros:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_3?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add|
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_4?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_2u
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_5?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_2y
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_6?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time?
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2_2/element_shape?
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2?
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensorn

zeros_like	ZerosLikelstm_cell_1/mul_6:z:0*
T0*'
_output_shapes
:?????????2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *%
_read_only_resource_inputs

*
bodyR
while_body_14163842*
condR
while_cond_14163841*`
output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????????????:??????????????????:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_14164360

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_1S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xg
mulMulmul/x:output:0strided_slice:output:0*
T0*#
_output_shapes
:?????????2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/xo
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*#
_output_shapes
:?????????2
mul_1H
ExpExpmul:z:0*
T0*#
_output_shapes
:?????????2
ExpW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?H2	
mul_2/x^
mul_2Mulmul_2/x:output:0Exp:y:0*
T0*#
_output_shapes
:?????????2
mul_2S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *9?@2
sub/yZ
subSub	mul_1:z:0sub/y:output:0*
T0*#
_output_shapes
:?????????2
subT
SigmoidSigmoidsub:z:0*
T0*#
_output_shapes
:?????????2	
SigmoidW
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2	
mul_3/xb
mul_3Mulmul_3/x:output:0Sigmoid:y:0*
T0*#
_output_shapes
:?????????2
mul_3|
stackPack	mul_2:z:0	mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*
axis?????????2
stackb
IdentityIdentitystack:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
lstm_1_while_body_14160462*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0i
elstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0<
8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0>
:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_06
2lstm_1_while_lstm_cell_1_readvariableop_resource_0
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5
lstm_1_while_identity_6'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorg
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor:
6lstm_1_while_lstm_cell_1_split_readvariableop_resource<
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource4
0lstm_1_while_lstm_cell_1_readvariableop_resource??'lstm_1/while/lstm_cell_1/ReadVariableOp?)lstm_1/while/lstm_cell_1/ReadVariableOp_1?)lstm_1/while/lstm_cell_1/ReadVariableOp_2?)lstm_1/while/lstm_cell_1/ReadVariableOp_3?-lstm_1/while/lstm_cell_1/split/ReadVariableOp?/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0lstm_1/while/TensorArrayV2Read/TensorListGetItem?
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape?
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0lstm_1_while_placeholderIlstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
24
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItem?
(lstm_1/while/lstm_cell_1/ones_like/ShapeShapelstm_1_while_placeholder_3*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/ones_like/Shape?
(lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(lstm_1/while/lstm_cell_1/ones_like/Const?
"lstm_1/while/lstm_cell_1/ones_likeFill1lstm_1/while/lstm_cell_1/ones_like/Shape:output:01lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/ones_like?
&lstm_1/while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2(
&lstm_1/while/lstm_cell_1/dropout/Const?
$lstm_1/while/lstm_cell_1/dropout/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:0/lstm_1/while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2&
$lstm_1/while/lstm_cell_1/dropout/Mul?
&lstm_1/while/lstm_cell_1/dropout/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell_1/dropout/Shape?
=lstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2?
=lstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniform?
/lstm_1/while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>21
/lstm_1/while/lstm_cell_1/dropout/GreaterEqual/y?
-lstm_1/while/lstm_cell_1/dropout/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2/
-lstm_1/while/lstm_cell_1/dropout/GreaterEqual?
%lstm_1/while/lstm_cell_1/dropout/CastCast1lstm_1/while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2'
%lstm_1/while/lstm_cell_1/dropout/Cast?
&lstm_1/while/lstm_cell_1/dropout/Mul_1Mul(lstm_1/while/lstm_cell_1/dropout/Mul:z:0)lstm_1/while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2(
&lstm_1/while/lstm_cell_1/dropout/Mul_1?
(lstm_1/while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2*
(lstm_1/while/lstm_cell_1/dropout_1/Const?
&lstm_1/while/lstm_cell_1/dropout_1/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2(
&lstm_1/while/lstm_cell_1/dropout_1/Mul?
(lstm_1/while/lstm_cell_1/dropout_1/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/dropout_1/Shape?
?lstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2A
?lstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
1lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>23
1lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/y?
/lstm_1/while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????21
/lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual?
'lstm_1/while/lstm_cell_1/dropout_1/CastCast3lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2)
'lstm_1/while/lstm_cell_1/dropout_1/Cast?
(lstm_1/while/lstm_cell_1/dropout_1/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_1/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2*
(lstm_1/while/lstm_cell_1/dropout_1/Mul_1?
(lstm_1/while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2*
(lstm_1/while/lstm_cell_1/dropout_2/Const?
&lstm_1/while/lstm_cell_1/dropout_2/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2(
&lstm_1/while/lstm_cell_1/dropout_2/Mul?
(lstm_1/while/lstm_cell_1/dropout_2/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/dropout_2/Shape?
?lstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2A
?lstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
1lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>23
1lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/y?
/lstm_1/while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????21
/lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual?
'lstm_1/while/lstm_cell_1/dropout_2/CastCast3lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2)
'lstm_1/while/lstm_cell_1/dropout_2/Cast?
(lstm_1/while/lstm_cell_1/dropout_2/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_2/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2*
(lstm_1/while/lstm_cell_1/dropout_2/Mul_1?
(lstm_1/while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2*
(lstm_1/while/lstm_cell_1/dropout_3/Const?
&lstm_1/while/lstm_cell_1/dropout_3/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2(
&lstm_1/while/lstm_cell_1/dropout_3/Mul?
(lstm_1/while/lstm_cell_1/dropout_3/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/dropout_3/Shape?
?lstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2A
?lstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
1lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>23
1lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/y?
/lstm_1/while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????21
/lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual?
'lstm_1/while/lstm_cell_1/dropout_3/CastCast3lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2)
'lstm_1/while/lstm_cell_1/dropout_3/Cast?
(lstm_1/while/lstm_cell_1/dropout_3/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_3/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2*
(lstm_1/while/lstm_cell_1/dropout_3/Mul_1?
lstm_1/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_1/while/lstm_cell_1/Const?
(lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_1/while/lstm_cell_1/split/split_dim?
-lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02/
-lstm_1/while/lstm_cell_1/split/ReadVariableOp?
lstm_1/while/lstm_cell_1/splitSplit1lstm_1/while/lstm_cell_1/split/split_dim:output:05lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2 
lstm_1/while/lstm_cell_1/split?
lstm_1/while/lstm_cell_1/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2!
lstm_1/while/lstm_cell_1/MatMul?
!lstm_1/while/lstm_cell_1/MatMul_1MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_1?
!lstm_1/while/lstm_cell_1/MatMul_2MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_2?
!lstm_1/while/lstm_cell_1/MatMul_3MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_3?
 lstm_1/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_1/while/lstm_cell_1/Const_1?
*lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_1/while/lstm_cell_1/split_1/split_dim?
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype021
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
 lstm_1/while/lstm_cell_1/split_1Split3lstm_1/while/lstm_cell_1/split_1/split_dim:output:07lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2"
 lstm_1/while/lstm_cell_1/split_1?
 lstm_1/while/lstm_cell_1/BiasAddBiasAdd)lstm_1/while/lstm_cell_1/MatMul:product:0)lstm_1/while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/while/lstm_cell_1/BiasAdd?
"lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd+lstm_1/while/lstm_cell_1/MatMul_1:product:0)lstm_1/while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_1?
"lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd+lstm_1/while/lstm_cell_1/MatMul_2:product:0)lstm_1/while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_2?
"lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd+lstm_1/while/lstm_cell_1/MatMul_3:product:0)lstm_1/while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/BiasAdd_3?
lstm_1/while/lstm_cell_1/mulMullstm_1_while_placeholder_3*lstm_1/while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/mul?
lstm_1/while/lstm_cell_1/mul_1Mullstm_1_while_placeholder_3,lstm_1/while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_1?
lstm_1/while/lstm_cell_1/mul_2Mullstm_1_while_placeholder_3,lstm_1/while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_2?
lstm_1/while/lstm_cell_1/mul_3Mullstm_1_while_placeholder_3,lstm_1/while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_3?
'lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02)
'lstm_1/while/lstm_cell_1/ReadVariableOp?
,lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_1/while/lstm_cell_1/strided_slice/stack?
.lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice/stack_1?
.lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell_1/strided_slice/stack_2?
&lstm_1/while/lstm_cell_1/strided_sliceStridedSlice/lstm_1/while/lstm_cell_1/ReadVariableOp:value:05lstm_1/while/lstm_cell_1/strided_slice/stack:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell_1/strided_slice?
!lstm_1/while/lstm_cell_1/MatMul_4MatMul lstm_1/while/lstm_cell_1/mul:z:0/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_4?
lstm_1/while/lstm_cell_1/addAddV2)lstm_1/while/lstm_cell_1/BiasAdd:output:0+lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/add?
 lstm_1/while/lstm_cell_1/SigmoidSigmoid lstm_1/while/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2"
 lstm_1/while/lstm_cell_1/Sigmoid?
)lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_1?
.lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice_1/stack?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:07lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_1?
!lstm_1/while/lstm_cell_1/MatMul_5MatMul"lstm_1/while/lstm_cell_1/mul_1:z:01lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_5?
lstm_1/while/lstm_cell_1/add_1AddV2+lstm_1/while/lstm_cell_1/BiasAdd_1:output:0+lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_1?
"lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid"lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/Sigmoid_1?
lstm_1/while/lstm_cell_1/mul_4Mul&lstm_1/while/lstm_cell_1/Sigmoid_1:y:0lstm_1_while_placeholder_4*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_4?
)lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_2?
.lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   20
.lstm_1/while/lstm_cell_1/strided_slice_2/stack?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:07lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_2?
!lstm_1/while/lstm_cell_1/MatMul_6MatMul"lstm_1/while/lstm_cell_1/mul_2:z:01lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_6?
lstm_1/while/lstm_cell_1/add_2AddV2+lstm_1/while/lstm_cell_1/BiasAdd_2:output:0+lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_2?
lstm_1/while/lstm_cell_1/TanhTanh"lstm_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/while/lstm_cell_1/Tanh?
lstm_1/while/lstm_cell_1/mul_5Mul$lstm_1/while/lstm_cell_1/Sigmoid:y:0!lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_5?
lstm_1/while/lstm_cell_1/add_3AddV2"lstm_1/while/lstm_cell_1/mul_4:z:0"lstm_1/while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_3?
)lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_3?
.lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   20
.lstm_1/while/lstm_cell_1/strided_slice_3/stack?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2?
(lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:07lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_3?
!lstm_1/while/lstm_cell_1/MatMul_7MatMul"lstm_1/while/lstm_cell_1/mul_3:z:01lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_1/while/lstm_cell_1/MatMul_7?
lstm_1/while/lstm_cell_1/add_4AddV2+lstm_1/while/lstm_cell_1/BiasAdd_3:output:0+lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/add_4?
"lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid"lstm_1/while/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2$
"lstm_1/while/lstm_cell_1/Sigmoid_2?
lstm_1/while/lstm_cell_1/Tanh_1Tanh"lstm_1/while/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2!
lstm_1/while/lstm_cell_1/Tanh_1?
lstm_1/while/lstm_cell_1/mul_6Mul&lstm_1/while/lstm_cell_1/Sigmoid_2:y:0#lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2 
lstm_1/while/lstm_cell_1/mul_6?
lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile/multiples?
lstm_1/while/TileTile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0$lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile?
lstm_1/while/SelectV2SelectV2lstm_1/while/Tile:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_2*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2?
lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_1/multiples?
lstm_1/while/Tile_1Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_1?
lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_2/multiples?
lstm_1/while/Tile_2Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:?????????2
lstm_1/while/Tile_2?
lstm_1/while/SelectV2_1SelectV2lstm_1/while/Tile_1:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_3*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2_1?
lstm_1/while/SelectV2_2SelectV2lstm_1/while/Tile_2:output:0"lstm_1/while/lstm_cell_1/add_3:z:0lstm_1_while_placeholder_4*
T0*'
_output_shapes
:?????????2
lstm_1/while/SelectV2_2?
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholderlstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype023
1lstm_1/while/TensorArrayV2Write/TensorListSetItemj
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add/y?
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/addn
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add_1/y?
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/add_1?
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity?
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_1?
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_2?
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_3?
lstm_1/while/Identity_4Identitylstm_1/while/SelectV2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_4?
lstm_1/while/Identity_5Identity lstm_1/while/SelectV2_1:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_5?
lstm_1/while/Identity_6Identity lstm_1/while/SelectV2_2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
lstm_1/while/Identity_6"7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0";
lstm_1_while_identity_6 lstm_1/while/Identity_6:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"f
0lstm_1_while_lstm_cell_1_readvariableop_resource2lstm_1_while_lstm_cell_1_readvariableop_resource_0"v
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_1_split_readvariableop_resource8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"?
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensorelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"?
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :?????????:?????????:?????????: : : :::2R
'lstm_1/while/lstm_cell_1/ReadVariableOp'lstm_1/while/lstm_cell_1/ReadVariableOp2V
)lstm_1/while/lstm_cell_1/ReadVariableOp_1)lstm_1/while/lstm_cell_1/ReadVariableOp_12V
)lstm_1/while/lstm_cell_1/ReadVariableOp_2)lstm_1/while/lstm_cell_1/ReadVariableOp_22V
)lstm_1/while/lstm_cell_1/ReadVariableOp_3)lstm_1/while/lstm_cell_1/ReadVariableOp_32^
-lstm_1/while/lstm_cell_1/split/ReadVariableOp-lstm_1/while/lstm_cell_1/split/ReadVariableOp2b
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
??
?
while_body_14162684
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/ones_like?
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2!
while/lstm_cell_1/dropout/Const?
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/dropout/Mul?
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_1/dropout/Shape?
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell_1/dropout/random_uniform/RandomUniform?
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2*
(while/lstm_cell_1/dropout/GreaterEqual/y?
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2(
&while/lstm_cell_1/dropout/GreaterEqual?
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2 
while/lstm_cell_1/dropout/Cast?
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout/Mul_1?
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_1/Const?
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_1/Mul?
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_1/Shape?
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_1/GreaterEqual/y?
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_1/GreaterEqual?
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_1/Cast?
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_1/Mul_1?
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_2/Const?
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_2/Mul?
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_2/Shape?
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2?P2:
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_2/GreaterEqual/y?
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_2/GreaterEqual?
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_2/Cast?
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_2/Mul_1?
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_3/Const?
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_3/Mul?
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_3/Shape?
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_3/GreaterEqual/y?
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_3/GreaterEqual?
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_3/Cast?
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_3/Mul_1t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mulMulwhile_placeholder_2#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mulwhile_placeholder_2%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mulwhile_placeholder_2%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mulwhile_placeholder_2%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_3?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_4?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_6?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_1/mul_6:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
??
?
D__inference_lstm_1_layer_call_and_return_conditional_losses_14163688

inputs
mask
-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????2
transpose_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
lstm_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/ones_likeh
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mulMulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_3?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add|
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_4?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_2u
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_5?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_2y
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_6?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time?
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2_2/element_shape?
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2?
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensorn

zeros_like	ZerosLikelstm_cell_1/mul_6:z:0*
T0*'
_output_shapes
:?????????2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *%
_read_only_resource_inputs

*
bodyR
while_body_14163541*
condR
while_cond_14163540*`
output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????????????:??????????????????:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
?
while_cond_14162683
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_14162683___redundant_placeholder06
2while_while_cond_14162683___redundant_placeholder16
2while_while_cond_14162683___redundant_placeholder26
2while_while_cond_14162683___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?
D__inference_lstm_1_layer_call_and_return_conditional_losses_14162299
inputs_0-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
lstm_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout/Const?
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Mul?
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shape?
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???22
0lstm_cell_1/dropout/random_uniform/RandomUniform?
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2$
"lstm_cell_1/dropout/GreaterEqual/y?
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_cell_1/dropout/GreaterEqual?
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Cast?
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_1/Const?
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Mul?
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shape?
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform?
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_1/GreaterEqual/y?
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_1/GreaterEqual?
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Cast?
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_2/Const?
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Mul?
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shape?
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2?ʡ24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform?
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_2/GreaterEqual/y?
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_2/GreaterEqual?
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Cast?
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_3/Const?
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Mul?
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shape?
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform?
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_3/GreaterEqual/y?
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_3/GreaterEqual?
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Cast?
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Mul_1h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mulMulzeros:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulzeros:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulzeros:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulzeros:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_3?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add|
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_4?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_2u
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_5?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_2y
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_6?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_14162138*
condR
while_cond_14162137*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?1
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14161918

inputs
assignmovingavg_14161893
assignmovingavg_1_14161899)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/14161893*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_14161893*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/14161893*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/14161893*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_14161893AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/14161893*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/14161899*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_14161899*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/14161899*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/14161899*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_14161899AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/14161899*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_1_layer_call_fn_14161994

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?k
?
.__inference_lstm_cell_1_layer_call_fn_14164705

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpZ
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ڪ'2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_3/Mul_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:P*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:P*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????2
	BiasAdd_3`
mulMulstates_0dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
mulf
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
mul_1f
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
mul_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
mul_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mul_4|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????2
Tanh^
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:P*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_6?
IdentityIdentity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
??
?
while_body_14162411
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/ones_liket
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mulMulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_3?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_4?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_6?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_1/mul_6:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
??
?
/__inference_sequential_1_layer_call_fn_14161827
masking_1_input;
7batch_normalization_1_batchnorm_readvariableop_resource?
;batch_normalization_1_batchnorm_mul_readvariableop_resource=
9batch_normalization_1_batchnorm_readvariableop_1_resource=
9batch_normalization_1_batchnorm_readvariableop_2_resource4
0lstm_1_lstm_cell_1_split_readvariableop_resource6
2lstm_1_lstm_cell_1_split_1_readvariableop_resource.
*lstm_1_lstm_cell_1_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??.batch_normalization_1/batchnorm/ReadVariableOp?0batch_normalization_1/batchnorm/ReadVariableOp_1?0batch_normalization_1/batchnorm/ReadVariableOp_2?2batch_normalization_1/batchnorm/mul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?!lstm_1/lstm_cell_1/ReadVariableOp?#lstm_1/lstm_cell_1/ReadVariableOp_1?#lstm_1/lstm_cell_1/ReadVariableOp_2?#lstm_1/lstm_cell_1/ReadVariableOp_3?'lstm_1/lstm_cell_1/split/ReadVariableOp?)lstm_1/lstm_cell_1/split_1/ReadVariableOp?lstm_1/whileq
masking_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
masking_1/NotEqual/y?
masking_1/NotEqualNotEqualmasking_1_inputmasking_1/NotEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
masking_1/NotEqual?
masking_1/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
masking_1/Any/reduction_indices?
masking_1/AnyAnymasking_1/NotEqual:z:0(masking_1/Any/reduction_indices:output:0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
masking_1/Any?
masking_1/CastCastmasking_1/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
masking_1/Cast?
masking_1/mulMulmasking_1_inputmasking_1/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
masking_1/mul?
masking_1/SqueezeSqueezemasking_1/Any:output:0*
T0
*0
_output_shapes
:??????????????????*
squeeze_dims

?????????2
masking_1/Squeeze?
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/Rsqrt?
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Mulmasking_1/mul:z:0'batch_normalization_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2'
%batch_normalization_1/batchnorm/mul_1?
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1?
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/mul_2?
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2?
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2'
%batch_normalization_1/batchnorm/add_1u
lstm_1/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
lstm_1/Shape?
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice/stack?
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_1?
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_2?
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slicej
lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros/mul/y?
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/mulm
lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros/Less/y?
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros/Lessp
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros/packed/1?
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros/packedm
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros/Const?
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/zerosn
lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros_1/mul/y?
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/mulq
lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_1/zeros_1/Less/y?
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_1/zeros_1/Lesst
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/zeros_1/packed/1?
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_1/zeros_1/packedq
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/zeros_1/Const?
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/zeros_1?
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose/perm?
lstm_1/transpose	Transpose)batch_normalization_1/batchnorm/add_1:z:0lstm_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_1/transposed
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:2
lstm_1/Shape_1?
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_1/stack?
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_1?
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_2?
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slice_1y
lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm_1/ExpandDims/dim?
lstm_1/ExpandDims
ExpandDimsmasking_1/Squeeze:output:0lstm_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????2
lstm_1/ExpandDims?
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_1/perm?
lstm_1/transpose_1	Transposelstm_1/ExpandDims:output:0 lstm_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????2
lstm_1/transpose_1?
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_1/TensorArrayV2/element_shape?
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2?
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_1/TensorArrayUnstack/TensorListFromTensor?
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_2/stack?
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_1?
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_2?
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_1/strided_slice_2?
"lstm_1/lstm_cell_1/ones_like/ShapeShapelstm_1/zeros:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/ones_like/Shape?
"lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"lstm_1/lstm_cell_1/ones_like/Const?
lstm_1/lstm_cell_1/ones_likeFill+lstm_1/lstm_cell_1/ones_like/Shape:output:0+lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/ones_likev
lstm_1/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const?
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_1/lstm_cell_1/split/split_dim?
'lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02)
'lstm_1/lstm_cell_1/split/ReadVariableOp?
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_1/lstm_cell_1/split?
lstm_1/lstm_cell_1/MatMulMatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul?
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_1?
lstm_1/lstm_cell_1/MatMul_2MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_2?
lstm_1/lstm_cell_1/MatMul_3MatMullstm_1/strided_slice_2:output:0!lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_3z
lstm_1/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const_1?
$lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_1/lstm_cell_1/split_1/split_dim?
)lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02+
)lstm_1/lstm_cell_1/split_1/ReadVariableOp?
lstm_1/lstm_cell_1/split_1Split-lstm_1/lstm_cell_1/split_1/split_dim:output:01lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_1/lstm_cell_1/split_1?
lstm_1/lstm_cell_1/BiasAddBiasAdd#lstm_1/lstm_cell_1/MatMul:product:0#lstm_1/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd?
lstm_1/lstm_cell_1/BiasAdd_1BiasAdd%lstm_1/lstm_cell_1/MatMul_1:product:0#lstm_1/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_1?
lstm_1/lstm_cell_1/BiasAdd_2BiasAdd%lstm_1/lstm_cell_1/MatMul_2:product:0#lstm_1/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_2?
lstm_1/lstm_cell_1/BiasAdd_3BiasAdd%lstm_1/lstm_cell_1/MatMul_3:product:0#lstm_1/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/BiasAdd_3?
lstm_1/lstm_cell_1/mulMullstm_1/zeros:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul?
lstm_1/lstm_cell_1/mul_1Mullstm_1/zeros:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_1?
lstm_1/lstm_cell_1/mul_2Mullstm_1/zeros:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_2?
lstm_1/lstm_cell_1/mul_3Mullstm_1/zeros:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_3?
!lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02#
!lstm_1/lstm_cell_1/ReadVariableOp?
&lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_1/lstm_cell_1/strided_slice/stack?
(lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice/stack_1?
(lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell_1/strided_slice/stack_2?
 lstm_1/lstm_cell_1/strided_sliceStridedSlice)lstm_1/lstm_cell_1/ReadVariableOp:value:0/lstm_1/lstm_cell_1/strided_slice/stack:output:01lstm_1/lstm_cell_1/strided_slice/stack_1:output:01lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2"
 lstm_1/lstm_cell_1/strided_slice?
lstm_1/lstm_cell_1/MatMul_4MatMullstm_1/lstm_cell_1/mul:z:0)lstm_1/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_4?
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/BiasAdd:output:0%lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add?
lstm_1/lstm_cell_1/SigmoidSigmoidlstm_1/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid?
#lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_1?
(lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice_1/stack?
*lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_1?
*lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_2?
"lstm_1/lstm_cell_1/strided_slice_1StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_1:value:01lstm_1/lstm_cell_1/strided_slice_1/stack:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_1?
lstm_1/lstm_cell_1/MatMul_5MatMullstm_1/lstm_cell_1/mul_1:z:0+lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_5?
lstm_1/lstm_cell_1/add_1AddV2%lstm_1/lstm_cell_1/BiasAdd_1:output:0%lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_1?
lstm_1/lstm_cell_1/Sigmoid_1Sigmoidlstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid_1?
lstm_1/lstm_cell_1/mul_4Mul lstm_1/lstm_cell_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_4?
#lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_2?
(lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2*
(lstm_1/lstm_cell_1/strided_slice_2/stack?
*lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_1?
*lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_2?
"lstm_1/lstm_cell_1/strided_slice_2StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_2:value:01lstm_1/lstm_cell_1/strided_slice_2/stack:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_2?
lstm_1/lstm_cell_1/MatMul_6MatMullstm_1/lstm_cell_1/mul_2:z:0+lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_6?
lstm_1/lstm_cell_1/add_2AddV2%lstm_1/lstm_cell_1/BiasAdd_2:output:0%lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_2?
lstm_1/lstm_cell_1/TanhTanhlstm_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Tanh?
lstm_1/lstm_cell_1/mul_5Mullstm_1/lstm_cell_1/Sigmoid:y:0lstm_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_5?
lstm_1/lstm_cell_1/add_3AddV2lstm_1/lstm_cell_1/mul_4:z:0lstm_1/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_3?
#lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_3?
(lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2*
(lstm_1/lstm_cell_1/strided_slice_3/stack?
*lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_1?
*lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_2?
"lstm_1/lstm_cell_1/strided_slice_3StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_3:value:01lstm_1/lstm_cell_1/strided_slice_3/stack:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_3?
lstm_1/lstm_cell_1/MatMul_7MatMullstm_1/lstm_cell_1/mul_3:z:0+lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/MatMul_7?
lstm_1/lstm_cell_1/add_4AddV2%lstm_1/lstm_cell_1/BiasAdd_3:output:0%lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/add_4?
lstm_1/lstm_cell_1/Sigmoid_2Sigmoidlstm_1/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Sigmoid_2?
lstm_1/lstm_cell_1/Tanh_1Tanhlstm_1/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/Tanh_1?
lstm_1/lstm_cell_1/mul_6Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_1/lstm_cell_1/mul_6?
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$lstm_1/TensorArrayV2_1/element_shape?
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2_1\
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/time?
$lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$lstm_1/TensorArrayV2_2/element_shape?
lstm_1/TensorArrayV2_2TensorListReserve-lstm_1/TensorArrayV2_2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
lstm_1/TensorArrayV2_2?
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape?
0lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm_1/transpose_1:y:0Glstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type022
0lstm_1/TensorArrayUnstack_1/TensorListFromTensor?
lstm_1/zeros_like	ZerosLikelstm_1/lstm_cell_1/mul_6:z:0*
T0*'
_output_shapes
:?????????2
lstm_1/zeros_like?
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_1/while/maximum_iterationsx
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/while/loop_counter?
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros_like:y:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:00lstm_1_lstm_cell_1_split_readvariableop_resource2lstm_1_lstm_cell_1_split_1_readvariableop_resource*lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *%
_read_only_resource_inputs

*&
bodyR
lstm_1_while_body_14161653*&
condR
lstm_1_while_cond_14161652*`
output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *
parallel_iterations 2
lstm_1/while?
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02+
)lstm_1/TensorArrayV2Stack/TensorListStack?
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_1/strided_slice_3/stack?
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_1/strided_slice_3/stack_1?
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_3/stack_2?
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_1/strided_slice_3?
lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_2/perm?
lstm_1/transpose_2	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_1/transpose_2t
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/runtime?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMullstm_1/strided_slice_3:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lambda_1/strided_slice/stack?
lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_1/strided_slice/stack_1?
lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_1/strided_slice/stack_2?
lambda_1/strided_sliceStridedSlicedense_1/BiasAdd:output:0%lambda_1/strided_slice/stack:output:0'lambda_1/strided_slice/stack_1:output:0'lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
lambda_1/strided_slice?
lambda_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_1/strided_slice_1/stack?
 lambda_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 lambda_1/strided_slice_1/stack_1?
 lambda_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 lambda_1/strided_slice_1/stack_2?
lambda_1/strided_slice_1StridedSlicedense_1/BiasAdd:output:0'lambda_1/strided_slice_1/stack:output:0)lambda_1/strided_slice_1/stack_1:output:0)lambda_1/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
lambda_1/strided_slice_1e
lambda_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda_1/mul/x?
lambda_1/mulMullambda_1/mul/x:output:0lambda_1/strided_slice:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/muli
lambda_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda_1/mul_1/x?
lambda_1/mul_1Mullambda_1/mul_1/x:output:0!lambda_1/strided_slice_1:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_1c
lambda_1/ExpExplambda_1/mul:z:0*
T0*#
_output_shapes
:?????????2
lambda_1/Expi
lambda_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *?g?H2
lambda_1/mul_2/x?
lambda_1/mul_2Mullambda_1/mul_2/x:output:0lambda_1/Exp:y:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_2e
lambda_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *9?@2
lambda_1/sub/y~
lambda_1/subSublambda_1/mul_1:z:0lambda_1/sub/y:output:0*
T0*#
_output_shapes
:?????????2
lambda_1/subo
lambda_1/SigmoidSigmoidlambda_1/sub:z:0*
T0*#
_output_shapes
:?????????2
lambda_1/Sigmoidi
lambda_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2
lambda_1/mul_3/x?
lambda_1/mul_3Mullambda_1/mul_3/x:output:0lambda_1/Sigmoid:y:0*
T0*#
_output_shapes
:?????????2
lambda_1/mul_3?
lambda_1/stackPacklambda_1/mul_2:z:0lambda_1/mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*
axis?????????2
lambda_1/stack?
IdentityIdentitylambda_1/stack:output:0/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:??????????????????:::::::::2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2F
!lstm_1/lstm_cell_1/ReadVariableOp!lstm_1/lstm_cell_1/ReadVariableOp2J
#lstm_1/lstm_cell_1/ReadVariableOp_1#lstm_1/lstm_cell_1/ReadVariableOp_12J
#lstm_1/lstm_cell_1/ReadVariableOp_2#lstm_1/lstm_cell_1/ReadVariableOp_22J
#lstm_1/lstm_cell_1/ReadVariableOp_3#lstm_1/lstm_cell_1/ReadVariableOp_32R
'lstm_1/lstm_cell_1/split/ReadVariableOp'lstm_1/lstm_cell_1/split/ReadVariableOp2V
)lstm_1/lstm_cell_1/split_1/ReadVariableOp)lstm_1/lstm_cell_1/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:e a
4
_output_shapes"
 :??????????????????
)
_user_specified_namemasking_1_input
?	
?
while_cond_14163841
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_16
2while_while_cond_14163841___redundant_placeholder06
2while_while_cond_14163841___redundant_placeholder16
2while_while_cond_14163841___redundant_placeholder26
2while_while_cond_14163841___redundant_placeholder36
2while_while_cond_14163841___redundant_placeholder4
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*j
_input_shapesY
W: : : : :?????????:?????????:?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
??
?
D__inference_lstm_1_layer_call_and_return_conditional_losses_14163419

inputs
mask
-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????2
transpose_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
lstm_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout/Const?
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Mul?
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shape?
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???22
0lstm_cell_1/dropout/random_uniform/RandomUniform?
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2$
"lstm_cell_1/dropout/GreaterEqual/y?
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_cell_1/dropout/GreaterEqual?
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Cast?
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_1/Const?
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Mul?
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shape?
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform?
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_1/GreaterEqual/y?
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_1/GreaterEqual?
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Cast?
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_2/Const?
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Mul?
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shape?
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform?
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_2/GreaterEqual/y?
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_2/GreaterEqual?
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Cast?
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
lstm_cell_1/dropout_3/Const?
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Mul?
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shape?
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform?
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$lstm_cell_1/dropout_3/GreaterEqual/y?
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2$
"lstm_cell_1/dropout_3/GreaterEqual?
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Cast?
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/dropout_3/Mul_1h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

:P*
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:P*
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mulMulzeros:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulzeros:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulzeros:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulzeros:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_3?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add|
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_4?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_2u
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_5?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:P*
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Sigmoid_2y
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_6?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time?
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2_2/element_shape?
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2?
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensorn

zeros_like	ZerosLikelstm_cell_1/mul_6:z:0*
T0*'
_output_shapes
:?????????2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *%
_read_only_resource_inputs

*
bodyR
while_body_14163240*
condR
while_cond_14163239*`
output_shapesO
M: : : : :?????????:?????????:?????????: : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????????????:??????????????????:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
??
?
while_body_14162138
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/ones_like?
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2!
while/lstm_cell_1/dropout/Const?
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/dropout/Mul?
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_1/dropout/Shape?
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???28
6while/lstm_cell_1/dropout/random_uniform/RandomUniform?
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2*
(while/lstm_cell_1/dropout/GreaterEqual/y?
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2(
&while/lstm_cell_1/dropout/GreaterEqual?
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2 
while/lstm_cell_1/dropout/Cast?
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout/Mul_1?
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_1/Const?
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_1/Mul?
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_1/Shape?
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_1/GreaterEqual/y?
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_1/GreaterEqual?
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_1/Cast?
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_1/Mul_1?
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_2/Const?
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_2/Mul?
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_2/Shape?
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_2/GreaterEqual/y?
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_2/GreaterEqual?
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_2/Cast?
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_2/Mul_1?
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2#
!while/lstm_cell_1/dropout_3/Const?
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????2!
while/lstm_cell_1/dropout_3/Mul?
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_3/Shape?
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2,
*while/lstm_cell_1/dropout_3/GreaterEqual/y?
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2*
(while/lstm_cell_1/dropout_3/GreaterEqual?
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2"
 while/lstm_cell_1/dropout_3/Cast?
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????2#
!while/lstm_cell_1/dropout_3/Mul_1t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

:P*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:P*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mulMulwhile_placeholder_2#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mulwhile_placeholder_2%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mulwhile_placeholder_2%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mulwhile_placeholder_2%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_3?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_4?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    <   2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    <   2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_6?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_1/mul_6:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
'sequential_1_lstm_1_while_cond_14157172D
@sequential_1_lstm_1_while_sequential_1_lstm_1_while_loop_counterJ
Fsequential_1_lstm_1_while_sequential_1_lstm_1_while_maximum_iterations)
%sequential_1_lstm_1_while_placeholder+
'sequential_1_lstm_1_while_placeholder_1+
'sequential_1_lstm_1_while_placeholder_2+
'sequential_1_lstm_1_while_placeholder_3+
'sequential_1_lstm_1_while_placeholder_4F
Bsequential_1_lstm_1_while_less_sequential_1_lstm_1_strided_slice_1^
Zsequential_1_lstm_1_while_sequential_1_lstm_1_while_cond_14157172___redundant_placeholder0^
Zsequential_1_lstm_1_while_sequential_1_lstm_1_while_cond_14157172___redundant_placeholder1^
Zsequential_1_lstm_1_while_sequential_1_lstm_1_while_cond_14157172___redundant_placeholder2^
Zsequential_1_lstm_1_while_sequential_1_lstm_1_while_cond_14157172___redundant_placeholder3^
Zsequential_1_lstm_1_while_sequential_1_lstm_1_while_cond_14157172___redundant_placeholder4&
"sequential_1_lstm_1_while_identity
?
sequential_1/lstm_1/while/LessLess%sequential_1_lstm_1_while_placeholderBsequential_1_lstm_1_while_less_sequential_1_lstm_1_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_1/lstm_1/while/Less?
"sequential_1/lstm_1/while/IdentityIdentity"sequential_1/lstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_1/lstm_1/while/Identity"Q
"sequential_1_lstm_1_while_identity+sequential_1/lstm_1/while/Identity:output:0*j
_input_shapesY
W: : : : :?????????:?????????:?????????: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:"?-
saver_filename:0
Identity:0Identity_318"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
X
masking_1_inputE
!serving_default_masking_1_input:0??????????????????<
lambda_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?q
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
l__call__
*m&call_and_return_all_conditional_losses
n_default_save_signature"?n
_tf_keras_sequential?n{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "masking_1_input"}}, {"class_name": "Masking", "config": {"name": "masking_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "mask_value": -99}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.25, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wUAAAAAAAAACQAAAAQAAABDAAAAc8YAAAB8AmQBawhzEHwCZAJrBHI2dACgAaEAZANrBHI2dACg\nAqEAZARrAnI2ZAV9BXQDoAR8BaEBAQB8BGQBawlyUGQGfQV0A6AEfAWhAQEAfAR9A3QFfACDAVwC\nfQZ9B3wDZAFrCXJ2fAN8BhQAfAN8BxQAAgB9Bn0HfAF0AKAGfAahARQAfQZ8AmQHawRyonQHoAh8\nAmQIGAChAX0IfAd8CBgAfQd8AnQAoAl8B6EBFAB9B3QAagp8BnwHZwJkCWQKjQJ9AHwAUwApC+Gi\nBwAARWxlbWVudHdpc2UgKExhbWJkYSkgY29tcHV0YXRpb24gb2YgYWxwaGEgYW5kIHJlZ3VsYXJp\nemVkIGJldGEuCgogICAgICAgIC0gQWxwaGE6CgogICAgICAgICAgICAoYWN0aXZhdGlvbikKICAg\nICAgICAgICAgRXhwb25lbnRpYWwgdW5pdHMgc2VlbXMgdG8gZ2l2ZSBmYXN0ZXIgdHJhaW5pbmcg\ndGhhbgogICAgICAgICAgICB0aGUgb3JpZ2luYWwgcGFwZXJzIHNvZnRwbHVzIHVuaXRzLiBNYWtl\ncyBzZW5zZSBkdWUgdG8gbG9nYXJpdGhtaWMKICAgICAgICAgICAgZWZmZWN0IG9mIGNoYW5nZSBp\nbiBhbHBoYS4KICAgICAgICAgICAgKGluaXRpYWxpemF0aW9uKQogICAgICAgICAgICBUbyBnZXQg\nZmFzdGVyIHRyYWluaW5nIGFuZCBmZXdlciBleHBsb2RpbmcgZ3JhZGllbnRzLAogICAgICAgICAg\nICBpbml0aWFsaXplIGFscGhhIHRvIGJlIGFyb3VuZCBpdHMgc2NhbGUgd2hlbiBiZXRhIGlzIGFy\nb3VuZCAxLjAsCiAgICAgICAgICAgIGFwcHJveCB0aGUgZXhwZWN0ZWQgdmFsdWUvbWVhbiBvZiB0\ncmFpbmluZyB0dGUuCiAgICAgICAgICAgIEJlY2F1c2Ugd2UncmUgbGF6eSB3ZSB3YW50IHRoZSBj\nb3JyZWN0IHNjYWxlIG9mIG91dHB1dCBidWlsdAogICAgICAgICAgICBpbnRvIHRoZSBtb2RlbCBz\nbyBpbml0aWFsaXplIGltcGxpY2l0bHk7CiAgICAgICAgICAgIG11bHRpcGx5IGFzc3VtZWQgZXhw\nKDApPTEgYnkgc2NhbGUgZmFjdG9yIGBpbml0X2FscGhhYC4KCiAgICAgICAgLSBCZXRhOgoKICAg\nICAgICAgICAgKGFjdGl2YXRpb24pCiAgICAgICAgICAgIFdlIHdhbnQgc2xvdyBjaGFuZ2VzIHdo\nZW4gYmV0YS0+IDAgc28gU29mdHBsdXMgbWFkZSBzZW5zZSBpbiB0aGUgb3JpZ2luYWwKICAgICAg\nICAgICAgcGFwZXIgYnV0IHdlIGdldCBzaW1pbGFyIGVmZmVjdCB3aXRoIHNpZ21vaWQuIEl0IGFs\nc28gaGFzIG5pY2UgZmVhdHVyZXMuCiAgICAgICAgICAgIChyZWd1bGFyaXphdGlvbikgVXNlIG1h\neF9iZXRhX3ZhbHVlIHRvIGltcGxpY2l0bHkgcmVndWxhcml6ZSB0aGUgbW9kZWwKICAgICAgICAg\nICAgKGluaXRpYWxpemF0aW9uKSBGaXhlZCB0byBiZWdpbiBtb3Zpbmcgc2xvd2x5IGFyb3VuZCAx\nLjAKCiAgICAgICAgLSBVc2FnZQogICAgICAgICAgICAuLiBjb2RlLWJsb2NrOjogcHl0aG9uCgog\nICAgICAgICAgICAgICAgbW9kZWwuYWRkKFRpbWVEaXN0cmlidXRlZChEZW5zZSgyKSkpCiAgICAg\nICAgICAgICAgICBtb2RlbC5hZGQoTGFtYmRhKHd0dGUub3V0cHV0X2xhbWJkYSwgYXJndW1lbnRz\nPXsiaW5pdF9hbHBoYSI6aW5pdF9hbHBoYSwgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAg\nICAgICAgICAgICAgICAgICAgICAgICAgICAgIm1heF9iZXRhX3ZhbHVlIjoyLjAKICAgICAgICAg\nICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pKQoKCiAgICAg\nICAgOnBhcmFtIHg6IHRlbnNvciB3aXRoIGxhc3QgZGltZW5zaW9uIGhhdmluZyBsZW5ndGggMiB3\naXRoIHhbLi4uLDBdID0gYWxwaGEsIHhbLi4uLDFdID0gYmV0YQogICAgICAgIDpwYXJhbSBpbml0\nX2FscGhhOiBpbml0aWFsIHZhbHVlIG9mIGBhbHBoYWAuIERlZmF1bHQgdmFsdWUgaXMgMS4wLgog\nICAgICAgIDpwYXJhbSBtYXhfYmV0YV92YWx1ZTogbWF4aW11bSBiZXRhIHZhbHVlLiBEZWZhdWx0\nIHZhbHVlIGlzIDUuMC4KICAgICAgICA6cGFyYW0gbWF4X2FscGhhX3ZhbHVlOiBtYXh1bXVtIGFs\ncGhhIHZhbHVlLiBEZWZhdWx0IGlzIGBOb25lYC4KICAgICAgICA6dHlwZSB4OiBBcnJheQogICAg\nICAgIDp0eXBlIGluaXRfYWxwaGE6IEZsb2F0CiAgICAgICAgOnR5cGUgbWF4X2JldGFfdmFsdWU6\nIEZsb2F0CiAgICAgICAgOnR5cGUgbWF4X2FscGhhX3ZhbHVlOiBGbG9hdAogICAgICAgIDpyZXR1\ncm4geDogQSBwb3NpdGl2ZSBgVGVuc29yYCBvZiBzYW1lIHNoYXBlIGFzIGlucHV0CiAgICAgICAg\nOnJ0eXBlOiBBcnJheQoKICAgIE7pAwAAAGdIr7ya8td6PtoKdGVuc29yZmxvd3rtICAgICAgICAg\nICAgVXNpbmcgdGVuc29yZmxvdyBiYWNrZW5kIGFuZCBhbGxvd2luZyBoaWdoIGBtYXhfYmV0YV92\nYWx1ZWAgbWF5IGxlYWQgdG8KICAgICAgICAgICAgZ3JhZGllbnQgTmFOIGR1cmluZyB0cmFpbmlu\nZyB1bmxlc3MgYEsuZXBzaWxvbigpYCBpcyBzbWFsbC4KICAgICAgICAgICAgQ2FsbCBga2VyYXMu\nYmFja2VuZC5zZXRfZXBzaWxvbigxZS0wOClgIHRvIGxvd2VyIGVwc2lsb24gICAgICAgICAgICAg\n+n9gYWxwaGFfa2VybmVsX3NjYWxlZmFjdG9yYCBkZXByZWNhdGVkIGluIGZhdm9yIG9mIGBzY2Fs\nZWZhY3RvcmAgc2NhbGluZyBib3RoLgogU2V0dGluZyBgc2NhbGVmYWN0b3IgPSBhbHBoYV9rZXJu\nZWxfc2NhbGVmYWN0b3JgZ83MzMzMzPA/ZwAAAAAAAPA/6f////8pAdoEYXhpcykL2gFL2gdlcHNp\nbG9u2gdiYWNrZW5k2gh3YXJuaW5nc9oEd2FybtoTX2tlcmFzX3Vuc3RhY2tfaGFja9oDZXhw2gJu\ncNoDbG9n2gdzaWdtb2lk2gVzdGFjaykJ2gF42gppbml0X2FscGhh2g5tYXhfYmV0YV92YWx1ZVoL\nc2NhbGVmYWN0b3LaGGFscGhhX2tlcm5lbF9zY2FsZWZhY3RvctoHbWVzc2FnZdoBYdoBYtoGX3No\naWZ0qQByGgAAAPozL3Vzci9sb2NhbC9saWIvcHl0aG9uMy43L2Rpc3QtcGFja2FnZXMvd3R0ZS93\ndHRlLnB52g1vdXRwdXRfbGFtYmRhHwAAAHMkAAAAAC8QARgGBAEKAQgBBAEKAQQCDAIIAhIDDgII\nAw4CCAIOAhIC\n", {"class_name": "__tuple__", "items": [1.0, 5.0, null, null]}, null]}, "function_type": "lambda", "module": "wtte.wtte", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {"init_alpha": 305981.4999972779, "max_beta_value": 100.0, "alpha_kernel_scalefactor": 0.5}}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "masking_1_input"}}, {"class_name": "Masking", "config": {"name": "masking_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "mask_value": -99}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.25, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wUAAAAAAAAACQAAAAQAAABDAAAAc8YAAAB8AmQBawhzEHwCZAJrBHI2dACgAaEAZANrBHI2dACg\nAqEAZARrAnI2ZAV9BXQDoAR8BaEBAQB8BGQBawlyUGQGfQV0A6AEfAWhAQEAfAR9A3QFfACDAVwC\nfQZ9B3wDZAFrCXJ2fAN8BhQAfAN8BxQAAgB9Bn0HfAF0AKAGfAahARQAfQZ8AmQHawRyonQHoAh8\nAmQIGAChAX0IfAd8CBgAfQd8AnQAoAl8B6EBFAB9B3QAagp8BnwHZwJkCWQKjQJ9AHwAUwApC+Gi\nBwAARWxlbWVudHdpc2UgKExhbWJkYSkgY29tcHV0YXRpb24gb2YgYWxwaGEgYW5kIHJlZ3VsYXJp\nemVkIGJldGEuCgogICAgICAgIC0gQWxwaGE6CgogICAgICAgICAgICAoYWN0aXZhdGlvbikKICAg\nICAgICAgICAgRXhwb25lbnRpYWwgdW5pdHMgc2VlbXMgdG8gZ2l2ZSBmYXN0ZXIgdHJhaW5pbmcg\ndGhhbgogICAgICAgICAgICB0aGUgb3JpZ2luYWwgcGFwZXJzIHNvZnRwbHVzIHVuaXRzLiBNYWtl\ncyBzZW5zZSBkdWUgdG8gbG9nYXJpdGhtaWMKICAgICAgICAgICAgZWZmZWN0IG9mIGNoYW5nZSBp\nbiBhbHBoYS4KICAgICAgICAgICAgKGluaXRpYWxpemF0aW9uKQogICAgICAgICAgICBUbyBnZXQg\nZmFzdGVyIHRyYWluaW5nIGFuZCBmZXdlciBleHBsb2RpbmcgZ3JhZGllbnRzLAogICAgICAgICAg\nICBpbml0aWFsaXplIGFscGhhIHRvIGJlIGFyb3VuZCBpdHMgc2NhbGUgd2hlbiBiZXRhIGlzIGFy\nb3VuZCAxLjAsCiAgICAgICAgICAgIGFwcHJveCB0aGUgZXhwZWN0ZWQgdmFsdWUvbWVhbiBvZiB0\ncmFpbmluZyB0dGUuCiAgICAgICAgICAgIEJlY2F1c2Ugd2UncmUgbGF6eSB3ZSB3YW50IHRoZSBj\nb3JyZWN0IHNjYWxlIG9mIG91dHB1dCBidWlsdAogICAgICAgICAgICBpbnRvIHRoZSBtb2RlbCBz\nbyBpbml0aWFsaXplIGltcGxpY2l0bHk7CiAgICAgICAgICAgIG11bHRpcGx5IGFzc3VtZWQgZXhw\nKDApPTEgYnkgc2NhbGUgZmFjdG9yIGBpbml0X2FscGhhYC4KCiAgICAgICAgLSBCZXRhOgoKICAg\nICAgICAgICAgKGFjdGl2YXRpb24pCiAgICAgICAgICAgIFdlIHdhbnQgc2xvdyBjaGFuZ2VzIHdo\nZW4gYmV0YS0+IDAgc28gU29mdHBsdXMgbWFkZSBzZW5zZSBpbiB0aGUgb3JpZ2luYWwKICAgICAg\nICAgICAgcGFwZXIgYnV0IHdlIGdldCBzaW1pbGFyIGVmZmVjdCB3aXRoIHNpZ21vaWQuIEl0IGFs\nc28gaGFzIG5pY2UgZmVhdHVyZXMuCiAgICAgICAgICAgIChyZWd1bGFyaXphdGlvbikgVXNlIG1h\neF9iZXRhX3ZhbHVlIHRvIGltcGxpY2l0bHkgcmVndWxhcml6ZSB0aGUgbW9kZWwKICAgICAgICAg\nICAgKGluaXRpYWxpemF0aW9uKSBGaXhlZCB0byBiZWdpbiBtb3Zpbmcgc2xvd2x5IGFyb3VuZCAx\nLjAKCiAgICAgICAgLSBVc2FnZQogICAgICAgICAgICAuLiBjb2RlLWJsb2NrOjogcHl0aG9uCgog\nICAgICAgICAgICAgICAgbW9kZWwuYWRkKFRpbWVEaXN0cmlidXRlZChEZW5zZSgyKSkpCiAgICAg\nICAgICAgICAgICBtb2RlbC5hZGQoTGFtYmRhKHd0dGUub3V0cHV0X2xhbWJkYSwgYXJndW1lbnRz\nPXsiaW5pdF9hbHBoYSI6aW5pdF9hbHBoYSwgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAg\nICAgICAgICAgICAgICAgICAgICAgICAgICAgIm1heF9iZXRhX3ZhbHVlIjoyLjAKICAgICAgICAg\nICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pKQoKCiAgICAg\nICAgOnBhcmFtIHg6IHRlbnNvciB3aXRoIGxhc3QgZGltZW5zaW9uIGhhdmluZyBsZW5ndGggMiB3\naXRoIHhbLi4uLDBdID0gYWxwaGEsIHhbLi4uLDFdID0gYmV0YQogICAgICAgIDpwYXJhbSBpbml0\nX2FscGhhOiBpbml0aWFsIHZhbHVlIG9mIGBhbHBoYWAuIERlZmF1bHQgdmFsdWUgaXMgMS4wLgog\nICAgICAgIDpwYXJhbSBtYXhfYmV0YV92YWx1ZTogbWF4aW11bSBiZXRhIHZhbHVlLiBEZWZhdWx0\nIHZhbHVlIGlzIDUuMC4KICAgICAgICA6cGFyYW0gbWF4X2FscGhhX3ZhbHVlOiBtYXh1bXVtIGFs\ncGhhIHZhbHVlLiBEZWZhdWx0IGlzIGBOb25lYC4KICAgICAgICA6dHlwZSB4OiBBcnJheQogICAg\nICAgIDp0eXBlIGluaXRfYWxwaGE6IEZsb2F0CiAgICAgICAgOnR5cGUgbWF4X2JldGFfdmFsdWU6\nIEZsb2F0CiAgICAgICAgOnR5cGUgbWF4X2FscGhhX3ZhbHVlOiBGbG9hdAogICAgICAgIDpyZXR1\ncm4geDogQSBwb3NpdGl2ZSBgVGVuc29yYCBvZiBzYW1lIHNoYXBlIGFzIGlucHV0CiAgICAgICAg\nOnJ0eXBlOiBBcnJheQoKICAgIE7pAwAAAGdIr7ya8td6PtoKdGVuc29yZmxvd3rtICAgICAgICAg\nICAgVXNpbmcgdGVuc29yZmxvdyBiYWNrZW5kIGFuZCBhbGxvd2luZyBoaWdoIGBtYXhfYmV0YV92\nYWx1ZWAgbWF5IGxlYWQgdG8KICAgICAgICAgICAgZ3JhZGllbnQgTmFOIGR1cmluZyB0cmFpbmlu\nZyB1bmxlc3MgYEsuZXBzaWxvbigpYCBpcyBzbWFsbC4KICAgICAgICAgICAgQ2FsbCBga2VyYXMu\nYmFja2VuZC5zZXRfZXBzaWxvbigxZS0wOClgIHRvIGxvd2VyIGVwc2lsb24gICAgICAgICAgICAg\n+n9gYWxwaGFfa2VybmVsX3NjYWxlZmFjdG9yYCBkZXByZWNhdGVkIGluIGZhdm9yIG9mIGBzY2Fs\nZWZhY3RvcmAgc2NhbGluZyBib3RoLgogU2V0dGluZyBgc2NhbGVmYWN0b3IgPSBhbHBoYV9rZXJu\nZWxfc2NhbGVmYWN0b3JgZ83MzMzMzPA/ZwAAAAAAAPA/6f////8pAdoEYXhpcykL2gFL2gdlcHNp\nbG9u2gdiYWNrZW5k2gh3YXJuaW5nc9oEd2FybtoTX2tlcmFzX3Vuc3RhY2tfaGFja9oDZXhw2gJu\ncNoDbG9n2gdzaWdtb2lk2gVzdGFjaykJ2gF42gppbml0X2FscGhh2g5tYXhfYmV0YV92YWx1ZVoL\nc2NhbGVmYWN0b3LaGGFscGhhX2tlcm5lbF9zY2FsZWZhY3RvctoHbWVzc2FnZdoBYdoBYtoGX3No\naWZ0qQByGgAAAPozL3Vzci9sb2NhbC9saWIvcHl0aG9uMy43L2Rpc3QtcGFja2FnZXMvd3R0ZS93\ndHRlLnB52g1vdXRwdXRfbGFtYmRhHwAAAHMkAAAAAC8QARgGBAEKAQgBBAEKAQQCDAIIAhIDDgII\nAw4CCAIOAhIC\n", {"class_name": "__tuple__", "items": [1.0, 5.0, null, null]}, null]}, "function_type": "lambda", "module": "wtte.wtte", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {"init_alpha": 305981.4999972779, "max_beta_value": 100.0, "alpha_kernel_scalefactor": 0.5}}}]}}, "training_config": {"loss": "loss_function", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipvalue": 0.5, "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
trainable_variables
	variables
regularization_losses
	keras_api
o__call__
*p&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Masking", "name": "masking_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "masking_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "mask_value": -99}}
?	
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
q__call__
*r&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1]}}
?
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
s__call__
*t&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.25, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1]}}
?

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
u__call__
*v&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?%
%trainable_variables
&	variables
'regularization_losses
(	keras_api
w__call__
*x&call_and_return_all_conditional_losses"?$
_tf_keras_layer?${"class_name": "Lambda", "name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wUAAAAAAAAACQAAAAQAAABDAAAAc8YAAAB8AmQBawhzEHwCZAJrBHI2dACgAaEAZANrBHI2dACg\nAqEAZARrAnI2ZAV9BXQDoAR8BaEBAQB8BGQBawlyUGQGfQV0A6AEfAWhAQEAfAR9A3QFfACDAVwC\nfQZ9B3wDZAFrCXJ2fAN8BhQAfAN8BxQAAgB9Bn0HfAF0AKAGfAahARQAfQZ8AmQHawRyonQHoAh8\nAmQIGAChAX0IfAd8CBgAfQd8AnQAoAl8B6EBFAB9B3QAagp8BnwHZwJkCWQKjQJ9AHwAUwApC+Gi\nBwAARWxlbWVudHdpc2UgKExhbWJkYSkgY29tcHV0YXRpb24gb2YgYWxwaGEgYW5kIHJlZ3VsYXJp\nemVkIGJldGEuCgogICAgICAgIC0gQWxwaGE6CgogICAgICAgICAgICAoYWN0aXZhdGlvbikKICAg\nICAgICAgICAgRXhwb25lbnRpYWwgdW5pdHMgc2VlbXMgdG8gZ2l2ZSBmYXN0ZXIgdHJhaW5pbmcg\ndGhhbgogICAgICAgICAgICB0aGUgb3JpZ2luYWwgcGFwZXJzIHNvZnRwbHVzIHVuaXRzLiBNYWtl\ncyBzZW5zZSBkdWUgdG8gbG9nYXJpdGhtaWMKICAgICAgICAgICAgZWZmZWN0IG9mIGNoYW5nZSBp\nbiBhbHBoYS4KICAgICAgICAgICAgKGluaXRpYWxpemF0aW9uKQogICAgICAgICAgICBUbyBnZXQg\nZmFzdGVyIHRyYWluaW5nIGFuZCBmZXdlciBleHBsb2RpbmcgZ3JhZGllbnRzLAogICAgICAgICAg\nICBpbml0aWFsaXplIGFscGhhIHRvIGJlIGFyb3VuZCBpdHMgc2NhbGUgd2hlbiBiZXRhIGlzIGFy\nb3VuZCAxLjAsCiAgICAgICAgICAgIGFwcHJveCB0aGUgZXhwZWN0ZWQgdmFsdWUvbWVhbiBvZiB0\ncmFpbmluZyB0dGUuCiAgICAgICAgICAgIEJlY2F1c2Ugd2UncmUgbGF6eSB3ZSB3YW50IHRoZSBj\nb3JyZWN0IHNjYWxlIG9mIG91dHB1dCBidWlsdAogICAgICAgICAgICBpbnRvIHRoZSBtb2RlbCBz\nbyBpbml0aWFsaXplIGltcGxpY2l0bHk7CiAgICAgICAgICAgIG11bHRpcGx5IGFzc3VtZWQgZXhw\nKDApPTEgYnkgc2NhbGUgZmFjdG9yIGBpbml0X2FscGhhYC4KCiAgICAgICAgLSBCZXRhOgoKICAg\nICAgICAgICAgKGFjdGl2YXRpb24pCiAgICAgICAgICAgIFdlIHdhbnQgc2xvdyBjaGFuZ2VzIHdo\nZW4gYmV0YS0+IDAgc28gU29mdHBsdXMgbWFkZSBzZW5zZSBpbiB0aGUgb3JpZ2luYWwKICAgICAg\nICAgICAgcGFwZXIgYnV0IHdlIGdldCBzaW1pbGFyIGVmZmVjdCB3aXRoIHNpZ21vaWQuIEl0IGFs\nc28gaGFzIG5pY2UgZmVhdHVyZXMuCiAgICAgICAgICAgIChyZWd1bGFyaXphdGlvbikgVXNlIG1h\neF9iZXRhX3ZhbHVlIHRvIGltcGxpY2l0bHkgcmVndWxhcml6ZSB0aGUgbW9kZWwKICAgICAgICAg\nICAgKGluaXRpYWxpemF0aW9uKSBGaXhlZCB0byBiZWdpbiBtb3Zpbmcgc2xvd2x5IGFyb3VuZCAx\nLjAKCiAgICAgICAgLSBVc2FnZQogICAgICAgICAgICAuLiBjb2RlLWJsb2NrOjogcHl0aG9uCgog\nICAgICAgICAgICAgICAgbW9kZWwuYWRkKFRpbWVEaXN0cmlidXRlZChEZW5zZSgyKSkpCiAgICAg\nICAgICAgICAgICBtb2RlbC5hZGQoTGFtYmRhKHd0dGUub3V0cHV0X2xhbWJkYSwgYXJndW1lbnRz\nPXsiaW5pdF9hbHBoYSI6aW5pdF9hbHBoYSwgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAg\nICAgICAgICAgICAgICAgICAgICAgICAgICAgIm1heF9iZXRhX3ZhbHVlIjoyLjAKICAgICAgICAg\nICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pKQoKCiAgICAg\nICAgOnBhcmFtIHg6IHRlbnNvciB3aXRoIGxhc3QgZGltZW5zaW9uIGhhdmluZyBsZW5ndGggMiB3\naXRoIHhbLi4uLDBdID0gYWxwaGEsIHhbLi4uLDFdID0gYmV0YQogICAgICAgIDpwYXJhbSBpbml0\nX2FscGhhOiBpbml0aWFsIHZhbHVlIG9mIGBhbHBoYWAuIERlZmF1bHQgdmFsdWUgaXMgMS4wLgog\nICAgICAgIDpwYXJhbSBtYXhfYmV0YV92YWx1ZTogbWF4aW11bSBiZXRhIHZhbHVlLiBEZWZhdWx0\nIHZhbHVlIGlzIDUuMC4KICAgICAgICA6cGFyYW0gbWF4X2FscGhhX3ZhbHVlOiBtYXh1bXVtIGFs\ncGhhIHZhbHVlLiBEZWZhdWx0IGlzIGBOb25lYC4KICAgICAgICA6dHlwZSB4OiBBcnJheQogICAg\nICAgIDp0eXBlIGluaXRfYWxwaGE6IEZsb2F0CiAgICAgICAgOnR5cGUgbWF4X2JldGFfdmFsdWU6\nIEZsb2F0CiAgICAgICAgOnR5cGUgbWF4X2FscGhhX3ZhbHVlOiBGbG9hdAogICAgICAgIDpyZXR1\ncm4geDogQSBwb3NpdGl2ZSBgVGVuc29yYCBvZiBzYW1lIHNoYXBlIGFzIGlucHV0CiAgICAgICAg\nOnJ0eXBlOiBBcnJheQoKICAgIE7pAwAAAGdIr7ya8td6PtoKdGVuc29yZmxvd3rtICAgICAgICAg\nICAgVXNpbmcgdGVuc29yZmxvdyBiYWNrZW5kIGFuZCBhbGxvd2luZyBoaWdoIGBtYXhfYmV0YV92\nYWx1ZWAgbWF5IGxlYWQgdG8KICAgICAgICAgICAgZ3JhZGllbnQgTmFOIGR1cmluZyB0cmFpbmlu\nZyB1bmxlc3MgYEsuZXBzaWxvbigpYCBpcyBzbWFsbC4KICAgICAgICAgICAgQ2FsbCBga2VyYXMu\nYmFja2VuZC5zZXRfZXBzaWxvbigxZS0wOClgIHRvIGxvd2VyIGVwc2lsb24gICAgICAgICAgICAg\n+n9gYWxwaGFfa2VybmVsX3NjYWxlZmFjdG9yYCBkZXByZWNhdGVkIGluIGZhdm9yIG9mIGBzY2Fs\nZWZhY3RvcmAgc2NhbGluZyBib3RoLgogU2V0dGluZyBgc2NhbGVmYWN0b3IgPSBhbHBoYV9rZXJu\nZWxfc2NhbGVmYWN0b3JgZ83MzMzMzPA/ZwAAAAAAAPA/6f////8pAdoEYXhpcykL2gFL2gdlcHNp\nbG9u2gdiYWNrZW5k2gh3YXJuaW5nc9oEd2FybtoTX2tlcmFzX3Vuc3RhY2tfaGFja9oDZXhw2gJu\ncNoDbG9n2gdzaWdtb2lk2gVzdGFjaykJ2gF42gppbml0X2FscGhh2g5tYXhfYmV0YV92YWx1ZVoL\nc2NhbGVmYWN0b3LaGGFscGhhX2tlcm5lbF9zY2FsZWZhY3RvctoHbWVzc2FnZdoBYdoBYtoGX3No\naWZ0qQByGgAAAPozL3Vzci9sb2NhbC9saWIvcHl0aG9uMy43L2Rpc3QtcGFja2FnZXMvd3R0ZS93\ndHRlLnB52g1vdXRwdXRfbGFtYmRhHwAAAHMkAAAAAC8QARgGBAEKAQgBBAEKAQQCDAIIAhIDDgII\nAw4CCAIOAhIC\n", {"class_name": "__tuple__", "items": [1.0, 5.0, null, null]}, null]}, "function_type": "lambda", "module": "wtte.wtte", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {"init_alpha": 305981.4999972779, "max_beta_value": 100.0, "alpha_kernel_scalefactor": 0.5}}}
?
)iter

*beta_1

+beta_2
	,decay
-learning_ratem^m_m` ma.mb/mc0mdvevfvg vh.vi/vj0vk"
	optimizer
Q
0
1
.2
/3
04
5
 6"
trackable_list_wrapper
_
0
1
2
3
.4
/5
06
7
 8"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
1non_trainable_variables
	variables
2metrics
	regularization_losses
3layer_metrics

4layers
5layer_regularization_losses
l__call__
n_default_save_signature
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
,
yserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
6non_trainable_variables
	variables
7metrics
regularization_losses
8layer_metrics

9layers
:layer_regularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
;non_trainable_variables
	variables
<metrics
regularization_losses
=layer_metrics

>layers
?layer_regularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
?

.kernel
/recurrent_kernel
0bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
z__call__
*{&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.25, "implementation": 1}}
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Dstates
trainable_variables
Enon_trainable_variables
	variables
Fmetrics
regularization_losses
Glayer_metrics

Hlayers
Ilayer_regularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
!trainable_variables
Jnon_trainable_variables
"	variables
Kmetrics
#regularization_losses
Llayer_metrics

Mlayers
Nlayer_regularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
%trainable_variables
Onon_trainable_variables
&	variables
Pmetrics
'regularization_losses
Qlayer_metrics

Rlayers
Slayer_regularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
+:)P2lstm_1/lstm_cell_1/kernel
5:3P2#lstm_1/lstm_cell_1/recurrent_kernel
%:#P2lstm_1/lstm_cell_1/bias
.
0
1"
trackable_list_wrapper
'
T0"
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@trainable_variables
Unon_trainable_variables
A	variables
Vmetrics
Bregularization_losses
Wlayer_metrics

Xlayers
Ylayer_regularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Ztotal
	[count
\	variables
]	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
Z0
[1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
.:,2"Adam/batch_normalization_1/gamma/m
-:+2!Adam/batch_normalization_1/beta/m
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
0:.P2 Adam/lstm_1/lstm_cell_1/kernel/m
::8P2*Adam/lstm_1/lstm_cell_1/recurrent_kernel/m
*:(P2Adam/lstm_1/lstm_cell_1/bias/m
.:,2"Adam/batch_normalization_1/gamma/v
-:+2!Adam/batch_normalization_1/beta/v
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
0:.P2 Adam/lstm_1/lstm_cell_1/kernel/v
::8P2*Adam/lstm_1/lstm_cell_1/recurrent_kernel/v
*:(P2Adam/lstm_1/lstm_cell_1/bias/v
?2?
/__inference_sequential_1_layer_call_fn_14161827
/__inference_sequential_1_layer_call_fn_14161509?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_1_layer_call_and_return_conditional_losses_14160792
J__inference_sequential_1_layer_call_and_return_conditional_losses_14161110?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_14157347?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *;?8
6?3
masking_1_input??????????????????
?2?
,__inference_masking_1_layer_call_fn_14161882?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_masking_1_layer_call_and_return_conditional_losses_14161871?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_1_layer_call_fn_14161974
8__inference_batch_normalization_1_layer_call_fn_14161994?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14161918
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14161938?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_lstm_1_layer_call_fn_14164021
)__inference_lstm_1_layer_call_fn_14163086
)__inference_lstm_1_layer_call_fn_14164290
)__inference_lstm_1_layer_call_fn_14162845?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_lstm_1_layer_call_and_return_conditional_losses_14163688
D__inference_lstm_1_layer_call_and_return_conditional_losses_14163419
D__inference_lstm_1_layer_call_and_return_conditional_losses_14162299
D__inference_lstm_1_layer_call_and_return_conditional_losses_14162540?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dense_1_layer_call_fn_14164310?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_14164300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_lambda_1_layer_call_fn_14164385
+__inference_lambda_1_layer_call_fn_14164410?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_lambda_1_layer_call_and_return_conditional_losses_14164360
F__inference_lambda_1_layer_call_and_return_conditional_losses_14164335?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_signature_wrapper_14161860masking_1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_lstm_cell_1_layer_call_fn_14164782
.__inference_lstm_cell_1_layer_call_fn_14164705?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_lstm_cell_1_layer_call_and_return_conditional_losses_14164519
I__inference_lstm_cell_1_layer_call_and_return_conditional_losses_14164596?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
#__inference__wrapped_model_14157347?	.0/ E?B
;?8
6?3
masking_1_input??????????????????
? "3?0
.
lambda_1"?
lambda_1??????????
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14161918|@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14161938|@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_1_layer_call_fn_14161974o@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
8__inference_batch_normalization_1_layer_call_fn_14161994o@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
E__inference_dense_1_layer_call_and_return_conditional_losses_14164300\ /?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
*__inference_dense_1_layer_call_fn_14164310O /?,
%?"
 ?
inputs?????????
? "???????????
F__inference_lambda_1_layer_call_and_return_conditional_losses_14164335`7?4
-?*
 ?
inputs?????????

 
p
? "%?"
?
0?????????
? ?
F__inference_lambda_1_layer_call_and_return_conditional_losses_14164360`7?4
-?*
 ?
inputs?????????

 
p 
? "%?"
?
0?????????
? ?
+__inference_lambda_1_layer_call_fn_14164385S7?4
-?*
 ?
inputs?????????

 
p
? "???????????
+__inference_lambda_1_layer_call_fn_14164410S7?4
-?*
 ?
inputs?????????

 
p 
? "???????????
D__inference_lstm_1_layer_call_and_return_conditional_losses_14162299}.0/O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????
? ?
D__inference_lstm_1_layer_call_and_return_conditional_losses_14162540}.0/O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????
? ?
D__inference_lstm_1_layer_call_and_return_conditional_losses_14163419?.0/m?j
c?`
-?*
inputs??????????????????
'?$
mask??????????????????

p

 
? "%?"
?
0?????????
? ?
D__inference_lstm_1_layer_call_and_return_conditional_losses_14163688?.0/m?j
c?`
-?*
inputs??????????????????
'?$
mask??????????????????

p 

 
? "%?"
?
0?????????
? ?
)__inference_lstm_1_layer_call_fn_14162845p.0/O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "???????????
)__inference_lstm_1_layer_call_fn_14163086p.0/O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "???????????
)__inference_lstm_1_layer_call_fn_14164021?.0/m?j
c?`
-?*
inputs??????????????????
'?$
mask??????????????????

p

 
? "???????????
)__inference_lstm_1_layer_call_fn_14164290?.0/m?j
c?`
-?*
inputs??????????????????
'?$
mask??????????????????

p 

 
? "???????????
I__inference_lstm_cell_1_layer_call_and_return_conditional_losses_14164519?.0/??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
I__inference_lstm_cell_1_layer_call_and_return_conditional_losses_14164596?.0/??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p 
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
.__inference_lstm_cell_1_layer_call_fn_14164705?.0/??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
.__inference_lstm_cell_1_layer_call_fn_14164782?.0/??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p 
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
G__inference_masking_1_layer_call_and_return_conditional_losses_14161871r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
,__inference_masking_1_layer_call_fn_14161882e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
J__inference_sequential_1_layer_call_and_return_conditional_losses_14160792?	.0/ M?J
C?@
6?3
masking_1_input??????????????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_14161110?	.0/ M?J
C?@
6?3
masking_1_input??????????????????
p 

 
? "%?"
?
0?????????
? ?
/__inference_sequential_1_layer_call_fn_14161509t	.0/ M?J
C?@
6?3
masking_1_input??????????????????
p

 
? "???????????
/__inference_sequential_1_layer_call_fn_14161827t	.0/ M?J
C?@
6?3
masking_1_input??????????????????
p 

 
? "???????????
&__inference_signature_wrapper_14161860?	.0/ X?U
? 
N?K
I
masking_1_input6?3
masking_1_input??????????????????"3?0
.
lambda_1"?
lambda_1?????????