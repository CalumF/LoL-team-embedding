╔╣

ф§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
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
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ьш
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ж@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	ж@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@ *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: @*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ж*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	@ж*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ж*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:ж*
dtype0
n
Adadelta/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdadelta/iter
g
!Adadelta/iter/Read/ReadVariableOpReadVariableOpAdadelta/iter*
_output_shapes
: *
dtype0	
p
Adadelta/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/decay
i
"Adadelta/decay/Read/ReadVariableOpReadVariableOpAdadelta/decay*
_output_shapes
: *
dtype0
ђ
Adadelta/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdadelta/learning_rate
y
*Adadelta/learning_rate/Read/ReadVariableOpReadVariableOpAdadelta/learning_rate*
_output_shapes
: *
dtype0
l
Adadelta/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/rho
e
 Adadelta/rho/Read/ReadVariableOpReadVariableOpAdadelta/rho*
_output_shapes
: *
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
Ю
 Adadelta/dense/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ж@*1
shared_name" Adadelta/dense/kernel/accum_grad
ќ
4Adadelta/dense/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense/kernel/accum_grad*
_output_shapes
:	ж@*
dtype0
ћ
Adadelta/dense/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adadelta/dense/bias/accum_grad
Ї
2Adadelta/dense/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/dense/bias/accum_grad*
_output_shapes
:@*
dtype0
а
"Adadelta/dense_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *3
shared_name$"Adadelta/dense_1/kernel/accum_grad
Ў
6Adadelta/dense_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_1/kernel/accum_grad*
_output_shapes

:@ *
dtype0
ў
 Adadelta/dense_1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adadelta/dense_1/bias/accum_grad
Љ
4Adadelta/dense_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_1/bias/accum_grad*
_output_shapes
: *
dtype0
а
"Adadelta/dense_2/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*3
shared_name$"Adadelta/dense_2/kernel/accum_grad
Ў
6Adadelta/dense_2/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_2/kernel/accum_grad*
_output_shapes

: @*
dtype0
ў
 Adadelta/dense_2/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adadelta/dense_2/bias/accum_grad
Љ
4Adadelta/dense_2/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_2/bias/accum_grad*
_output_shapes
:@*
dtype0
А
"Adadelta/dense_3/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ж*3
shared_name$"Adadelta/dense_3/kernel/accum_grad
џ
6Adadelta/dense_3/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_3/kernel/accum_grad*
_output_shapes
:	@ж*
dtype0
Ў
 Adadelta/dense_3/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:ж*1
shared_name" Adadelta/dense_3/bias/accum_grad
њ
4Adadelta/dense_3/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_3/bias/accum_grad*
_output_shapes	
:ж*
dtype0
Џ
Adadelta/dense/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ж@*0
shared_name!Adadelta/dense/kernel/accum_var
ћ
3Adadelta/dense/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/kernel/accum_var*
_output_shapes
:	ж@*
dtype0
њ
Adadelta/dense/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdadelta/dense/bias/accum_var
І
1Adadelta/dense/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/bias/accum_var*
_output_shapes
:@*
dtype0
ъ
!Adadelta/dense_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *2
shared_name#!Adadelta/dense_1/kernel/accum_var
Ќ
5Adadelta/dense_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_1/kernel/accum_var*
_output_shapes

:@ *
dtype0
ќ
Adadelta/dense_1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adadelta/dense_1/bias/accum_var
Ј
3Adadelta/dense_1/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_1/bias/accum_var*
_output_shapes
: *
dtype0
ъ
!Adadelta/dense_2/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*2
shared_name#!Adadelta/dense_2/kernel/accum_var
Ќ
5Adadelta/dense_2/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_2/kernel/accum_var*
_output_shapes

: @*
dtype0
ќ
Adadelta/dense_2/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adadelta/dense_2/bias/accum_var
Ј
3Adadelta/dense_2/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_2/bias/accum_var*
_output_shapes
:@*
dtype0
Ъ
!Adadelta/dense_3/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ж*2
shared_name#!Adadelta/dense_3/kernel/accum_var
ў
5Adadelta/dense_3/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_3/kernel/accum_var*
_output_shapes
:	@ж*
dtype0
Ќ
Adadelta/dense_3/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:ж*0
shared_name!Adadelta/dense_3/bias/accum_var
љ
3Adadelta/dense_3/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_3/bias/accum_var*
_output_shapes	
:ж*
dtype0

NoOpNoOp
 .
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*║.
value░.BГ. Bд.
џ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
╔
$iter
	%decay
&learning_rate
'rho
accum_gradF
accum_gradG
accum_gradH
accum_gradI
accum_gradJ
accum_gradK
accum_gradL
accum_gradM	accum_varN	accum_varO	accum_varP	accum_varQ	accum_varR	accum_varS	accum_varT	accum_varU
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
Г
(layer_regularization_losses
	variables
regularization_losses

)layers
*metrics
	trainable_variables
+non_trainable_variables
,layer_metrics
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
-layer_regularization_losses
	variables
regularization_losses

.layers
/metrics
trainable_variables
0non_trainable_variables
1layer_metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
2layer_regularization_losses
	variables
regularization_losses

3layers
4metrics
trainable_variables
5non_trainable_variables
6layer_metrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
7layer_regularization_losses
	variables
regularization_losses

8layers
9metrics
trainable_variables
:non_trainable_variables
;layer_metrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
<layer_regularization_losses
 	variables
!regularization_losses

=layers
>metrics
"trainable_variables
?non_trainable_variables
@layer_metrics
LJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4

A0
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
	Btotal
	Ccount
D	variables
E	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

D	variables
њЈ
VARIABLE_VALUE Adadelta/dense/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUEAdadelta/dense/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
ћЉ
VARIABLE_VALUE"Adadelta/dense_1/kernel/accum_grad[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE Adadelta/dense_1/bias/accum_gradYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
ћЉ
VARIABLE_VALUE"Adadelta/dense_2/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE Adadelta/dense_2/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
ћЉ
VARIABLE_VALUE"Adadelta/dense_3/kernel/accum_grad[layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE Adadelta/dense_3/bias/accum_gradYlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUEAdadelta/dense/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUEAdadelta/dense/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
њЈ
VARIABLE_VALUE!Adadelta/dense_1/kernel/accum_varZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUEAdadelta/dense_1/bias/accum_varXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
њЈ
VARIABLE_VALUE!Adadelta/dense_2/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUEAdadelta/dense_2/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
њЈ
VARIABLE_VALUE!Adadelta/dense_3/kernel/accum_varZlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUEAdadelta/dense_3/bias/accum_varXlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:         ж*
dtype0*
shape:         ж
Ќ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*(
_output_shapes
:         ж**
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*/
f*R(
&__inference_signature_wrapper_27251305
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
п
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adadelta/dense/kernel/accum_grad/Read/ReadVariableOp2Adadelta/dense/bias/accum_grad/Read/ReadVariableOp6Adadelta/dense_1/kernel/accum_grad/Read/ReadVariableOp4Adadelta/dense_1/bias/accum_grad/Read/ReadVariableOp6Adadelta/dense_2/kernel/accum_grad/Read/ReadVariableOp4Adadelta/dense_2/bias/accum_grad/Read/ReadVariableOp6Adadelta/dense_3/kernel/accum_grad/Read/ReadVariableOp4Adadelta/dense_3/bias/accum_grad/Read/ReadVariableOp3Adadelta/dense/kernel/accum_var/Read/ReadVariableOp1Adadelta/dense/bias/accum_var/Read/ReadVariableOp5Adadelta/dense_1/kernel/accum_var/Read/ReadVariableOp3Adadelta/dense_1/bias/accum_var/Read/ReadVariableOp5Adadelta/dense_2/kernel/accum_var/Read/ReadVariableOp3Adadelta/dense_2/bias/accum_var/Read/ReadVariableOp5Adadelta/dense_3/kernel/accum_var/Read/ReadVariableOp3Adadelta/dense_3/bias/accum_var/Read/ReadVariableOpConst*+
Tin$
"2 	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_save_27251737
ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcount Adadelta/dense/kernel/accum_gradAdadelta/dense/bias/accum_grad"Adadelta/dense_1/kernel/accum_grad Adadelta/dense_1/bias/accum_grad"Adadelta/dense_2/kernel/accum_grad Adadelta/dense_2/bias/accum_grad"Adadelta/dense_3/kernel/accum_grad Adadelta/dense_3/bias/accum_gradAdadelta/dense/kernel/accum_varAdadelta/dense/bias/accum_var!Adadelta/dense_1/kernel/accum_varAdadelta/dense_1/bias/accum_var!Adadelta/dense_2/kernel/accum_varAdadelta/dense_2/bias/accum_var!Adadelta/dense_3/kernel/accum_varAdadelta/dense_3/bias/accum_var**
Tin#
!2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference__traced_restore_27251839╗Ь
ќ`
Ы
C__inference_model_layer_call_and_return_conditional_losses_27251459

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3ѕа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ж@*
dtype02
dense/MatMul/ReadVariableOpЁ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2

dense/ReluЉ
dense/ActivityRegularizer/AbsAbsdense/Relu:activations:0*
T0*'
_output_shapes
:         @2
dense/ActivityRegularizer/AbsЊ
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense/ActivityRegularizer/Const│
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/SumЄ
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82!
dense/ActivityRegularizer/mul/xИ
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/mulЄ
dense/ActivityRegularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense/ActivityRegularizer/add/xх
dense/ActivityRegularizer/addAddV2(dense/ActivityRegularizer/add/x:output:0!dense/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/addі
dense/ActivityRegularizer/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shapeе
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackг
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1г
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2■
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceф
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast╣
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/add:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivЦ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_1/ReluЌ
dense_1/ActivityRegularizer/AbsAbsdense_1/Relu:activations:0*
T0*'
_output_shapes
:          2!
dense_1/ActivityRegularizer/AbsЌ
!dense_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_1/ActivityRegularizer/Const╗
dense_1/ActivityRegularizer/SumSum#dense_1/ActivityRegularizer/Abs:y:0*dense_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/SumІ
!dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82#
!dense_1/ActivityRegularizer/mul/x└
dense_1/ActivityRegularizer/mulMul*dense_1/ActivityRegularizer/mul/x:output:0(dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/mulІ
!dense_1/ActivityRegularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_1/ActivityRegularizer/add/xй
dense_1/ActivityRegularizer/addAddV2*dense_1/ActivityRegularizer/add/x:output:0#dense_1/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/addљ
!dense_1/ActivityRegularizer/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shapeг
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack░
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1░
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2і
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice░
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Cast┴
#dense_1/ActivityRegularizer/truedivRealDiv#dense_1/ActivityRegularizer/add:z:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivЦ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02
dense_2/MatMul/ReadVariableOpЪ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_2/MatMulц
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOpА
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
dense_2/ReluЌ
dense_2/ActivityRegularizer/AbsAbsdense_2/Relu:activations:0*
T0*'
_output_shapes
:         @2!
dense_2/ActivityRegularizer/AbsЌ
!dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_2/ActivityRegularizer/Const╗
dense_2/ActivityRegularizer/SumSum#dense_2/ActivityRegularizer/Abs:y:0*dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/SumІ
!dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82#
!dense_2/ActivityRegularizer/mul/x└
dense_2/ActivityRegularizer/mulMul*dense_2/ActivityRegularizer/mul/x:output:0(dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/mulІ
!dense_2/ActivityRegularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_2/ActivityRegularizer/add/xй
dense_2/ActivityRegularizer/addAddV2*dense_2/ActivityRegularizer/add/x:output:0#dense_2/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/addљ
!dense_2/ActivityRegularizer/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shapeг
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack░
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1░
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2і
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice░
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast┴
#dense_2/ActivityRegularizer/truedivRealDiv#dense_2/ActivityRegularizer/add:z:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truedivд
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@ж*
dtype02
dense_3/MatMul/ReadVariableOpа
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ж2
dense_3/MatMulЦ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:ж*
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ж2
dense_3/BiasAddz
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         ж2
dense_3/Sigmoidh
IdentityIdentitydense_3/Sigmoid:y:0*
T0*(
_output_shapes
:         ж2

Identityl

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2

Identity_1n

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2

Identity_2n

Identity_3Identity'dense_2/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*G
_input_shapes6
4:         ж:::::::::P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ш	
О
(__inference_model_layer_call_fn_27251507

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*.
_output_shapes
:         ж: : : **
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_272512542
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ш	
О
(__inference_model_layer_call_fn_27251483

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*.
_output_shapes
:         ж: : : **
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_272511792
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
у
Ф
C__inference_dense_layer_call_and_return_conditional_losses_27250913

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ж@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ж:::P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
у
Ф
C__inference_dense_layer_call_and_return_conditional_losses_27251518

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ж@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ж:::P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Т
Г
E__inference_dense_1_layer_call_and_return_conditional_losses_27250960

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
║H
─
C__inference_model_layer_call_and_return_conditional_losses_27251254

inputs
dense_27251206
dense_27251208
dense_1_27251219
dense_1_27251221
dense_2_27251232
dense_2_27251234
dense_3_27251245
dense_3_27251247
identity

identity_1

identity_2

identity_3ѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallж
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_27251206dense_27251208*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_272509132
dense/StatefulPartitionedCall╬
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*8
f3R1
/__inference_dense_activity_regularizer_272508682+
)dense/ActivityRegularizer/PartitionedCallў
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shapeе
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackг
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1г
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2■
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceф
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast╩
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivЊ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_27251219dense_1_27251221*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_272509602!
dense_1/StatefulPartitionedCallо
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*:
f5R3
1__inference_dense_1_activity_regularizer_272508832-
+dense_1/ActivityRegularizer/PartitionedCallъ
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shapeг
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack░
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1░
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2і
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice░
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Castм
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivЋ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_27251232dense_2_27251234*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_272510072!
dense_2/StatefulPartitionedCallо
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*:
f5R3
1__inference_dense_2_activity_regularizer_272508982-
+dense_2/ActivityRegularizer/PartitionedCallъ
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shapeг
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack░
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1░
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2і
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice░
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Castм
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truedivќ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_27251245dense_3_27251247*
Tin
2*
Tout
2*(
_output_shapes
:         ж*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_272510542!
dense_3/StatefulPartitionedCallЃ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*(
_output_shapes
:         ж2

IdentityЫ

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1З

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2З

Identity_3Identity'dense_2/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*G
_input_shapes6
4:         ж::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Щ

*__inference_dense_3_layer_call_fn_27251620

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         ж*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_272510542
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
║H
─
C__inference_model_layer_call_and_return_conditional_losses_27251179

inputs
dense_27251131
dense_27251133
dense_1_27251144
dense_1_27251146
dense_2_27251157
dense_2_27251159
dense_3_27251170
dense_3_27251172
identity

identity_1

identity_2

identity_3ѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallж
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_27251131dense_27251133*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_272509132
dense/StatefulPartitionedCall╬
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*8
f3R1
/__inference_dense_activity_regularizer_272508682+
)dense/ActivityRegularizer/PartitionedCallў
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shapeе
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackг
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1г
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2■
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceф
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast╩
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivЊ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_27251144dense_1_27251146*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_272509602!
dense_1/StatefulPartitionedCallо
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*:
f5R3
1__inference_dense_1_activity_regularizer_272508832-
+dense_1/ActivityRegularizer/PartitionedCallъ
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shapeг
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack░
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1░
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2і
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice░
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Castм
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivЋ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_27251157dense_2_27251159*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_272510072!
dense_2/StatefulPartitionedCallо
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*:
f5R3
1__inference_dense_2_activity_regularizer_272508982-
+dense_2/ActivityRegularizer/PartitionedCallъ
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shapeг
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack░
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1░
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2і
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice░
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Castм
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truedivќ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_27251170dense_3_27251172*
Tin
2*
Tout
2*(
_output_shapes
:         ж*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_272510542!
dense_3/StatefulPartitionedCallЃ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*(
_output_shapes
:         ж2

IdentityЫ

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1З

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2З

Identity_3Identity'dense_2/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*G
_input_shapes6
4:         ж::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
яѕ
ў
$__inference__traced_restore_27251839
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias$
 assignvariableop_8_adadelta_iter%
!assignvariableop_9_adadelta_decay.
*assignvariableop_10_adadelta_learning_rate$
 assignvariableop_11_adadelta_rho
assignvariableop_12_total
assignvariableop_13_count8
4assignvariableop_14_adadelta_dense_kernel_accum_grad6
2assignvariableop_15_adadelta_dense_bias_accum_grad:
6assignvariableop_16_adadelta_dense_1_kernel_accum_grad8
4assignvariableop_17_adadelta_dense_1_bias_accum_grad:
6assignvariableop_18_adadelta_dense_2_kernel_accum_grad8
4assignvariableop_19_adadelta_dense_2_bias_accum_grad:
6assignvariableop_20_adadelta_dense_3_kernel_accum_grad8
4assignvariableop_21_adadelta_dense_3_bias_accum_grad7
3assignvariableop_22_adadelta_dense_kernel_accum_var5
1assignvariableop_23_adadelta_dense_bias_accum_var9
5assignvariableop_24_adadelta_dense_1_kernel_accum_var7
3assignvariableop_25_adadelta_dense_1_bias_accum_var9
5assignvariableop_26_adadelta_dense_2_kernel_accum_var7
3assignvariableop_27_adadelta_dense_2_bias_accum_var9
5assignvariableop_28_adadelta_dense_3_kernel_accum_var7
3assignvariableop_29_adadelta_dense_3_bias_accum_var
identity_31ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1џ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*д
valueюBЎB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names╩
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices┬
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ї
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЇ
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Њ
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ќ
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ћ
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ќ
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ћ
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ќ
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ћ
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8ќ
AssignVariableOp_8AssignVariableOp assignvariableop_8_adadelta_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ќ
AssignVariableOp_9AssignVariableOp!assignvariableop_9_adadelta_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Б
AssignVariableOp_10AssignVariableOp*assignvariableop_10_adadelta_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ў
AssignVariableOp_11AssignVariableOp assignvariableop_11_adadelta_rhoIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12њ
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13њ
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Г
AssignVariableOp_14AssignVariableOp4assignvariableop_14_adadelta_dense_kernel_accum_gradIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ф
AssignVariableOp_15AssignVariableOp2assignvariableop_15_adadelta_dense_bias_accum_gradIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16»
AssignVariableOp_16AssignVariableOp6assignvariableop_16_adadelta_dense_1_kernel_accum_gradIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Г
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adadelta_dense_1_bias_accum_gradIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18»
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adadelta_dense_2_kernel_accum_gradIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Г
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adadelta_dense_2_bias_accum_gradIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20»
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adadelta_dense_3_kernel_accum_gradIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Г
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adadelta_dense_3_bias_accum_gradIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22г
AssignVariableOp_22AssignVariableOp3assignvariableop_22_adadelta_dense_kernel_accum_varIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23ф
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adadelta_dense_bias_accum_varIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24«
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adadelta_dense_1_kernel_accum_varIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25г
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adadelta_dense_1_bias_accum_varIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26«
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adadelta_dense_2_kernel_accum_varIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27г
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adadelta_dense_2_bias_accum_varIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28«
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adadelta_dense_3_kernel_accum_varIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29г
AssignVariableOp_29AssignVariableOp3assignvariableop_29_adadelta_dense_3_bias_accum_varIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29е
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesћ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЫ
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_30 
Identity_31IdentityIdentity_30:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_31"#
identity_31Identity_31:output:0*Ї
_input_shapes|
z: ::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
▀

«
I__inference_dense_2_layer_call_and_return_all_conditional_losses_27251600

inputs
unknown
	unknown_0
identity

identity_1ѕбStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_272510072
StatefulPartitionedCallќ
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*:
f5R3
1__inference_dense_2_activity_regularizer_272508982
PartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Т
Г
E__inference_dense_2_layer_call_and_return_conditional_losses_27251007

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Э

*__inference_dense_1_layer_call_fn_27251558

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_272509602
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ш
}
(__inference_dense_layer_call_fn_27251527

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_272509132
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ж::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
▀

«
I__inference_dense_1_layer_call_and_return_all_conditional_losses_27251569

inputs
unknown
	unknown_0
identity

identity_1ѕбStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_272509602
StatefulPartitionedCallќ
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*:
f5R3
1__inference_dense_1_activity_regularizer_272508832
PartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
їO
Ї
!__inference__traced_save_27251737
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adadelta_dense_kernel_accum_grad_read_readvariableop=
9savev2_adadelta_dense_bias_accum_grad_read_readvariableopA
=savev2_adadelta_dense_1_kernel_accum_grad_read_readvariableop?
;savev2_adadelta_dense_1_bias_accum_grad_read_readvariableopA
=savev2_adadelta_dense_2_kernel_accum_grad_read_readvariableop?
;savev2_adadelta_dense_2_bias_accum_grad_read_readvariableopA
=savev2_adadelta_dense_3_kernel_accum_grad_read_readvariableop?
;savev2_adadelta_dense_3_bias_accum_grad_read_readvariableop>
:savev2_adadelta_dense_kernel_accum_var_read_readvariableop<
8savev2_adadelta_dense_bias_accum_var_read_readvariableop@
<savev2_adadelta_dense_1_kernel_accum_var_read_readvariableop>
:savev2_adadelta_dense_1_bias_accum_var_read_readvariableop@
<savev2_adadelta_dense_2_kernel_accum_var_read_readvariableop>
:savev2_adadelta_dense_2_bias_accum_var_read_readvariableop@
<savev2_adadelta_dense_3_kernel_accum_var_read_readvariableop>
:savev2_adadelta_dense_3_bias_accum_var_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1Ј
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_02683aa9b63140a1b83ff704a31cec4a/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameћ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*д
valueюBЎB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names─
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices┌
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adadelta_dense_kernel_accum_grad_read_readvariableop9savev2_adadelta_dense_bias_accum_grad_read_readvariableop=savev2_adadelta_dense_1_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_1_bias_accum_grad_read_readvariableop=savev2_adadelta_dense_2_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_2_bias_accum_grad_read_readvariableop=savev2_adadelta_dense_3_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_3_bias_accum_grad_read_readvariableop:savev2_adadelta_dense_kernel_accum_var_read_readvariableop8savev2_adadelta_dense_bias_accum_var_read_readvariableop<savev2_adadelta_dense_1_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_1_bias_accum_var_read_readvariableop<savev2_adadelta_dense_2_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_2_bias_accum_var_read_readvariableop<savev2_adadelta_dense_3_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_3_bias_accum_var_read_readvariableop"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2Ѓ
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardг
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1б
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesј
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices¤
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1с
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesг
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ь
_input_shapes▄
┘: :	ж@:@:@ : : @:@:	@ж:ж: : : : : : :	ж@:@:@ : : @:@:	@ж:ж:	ж@:@:@ : : @:@:	@ж:ж: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ж@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:%!

_output_shapes
:	@ж:!

_output_shapes	
:ж:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ж@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:%!

_output_shapes
:	@ж:!

_output_shapes	
:ж:%!

_output_shapes
:	ж@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:%!

_output_shapes
:	@ж:!

_output_shapes	
:ж:

_output_shapes
: 
щ	
п
(__inference_model_layer_call_fn_27251276
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*.
_output_shapes
:         ж: : : **
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_272512542
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ж
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Т
Г
E__inference_dense_2_layer_call_and_return_conditional_losses_27251580

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
йH
┼
C__inference_model_layer_call_and_return_conditional_losses_27251125
input_1
dense_27251077
dense_27251079
dense_1_27251090
dense_1_27251092
dense_2_27251103
dense_2_27251105
dense_3_27251116
dense_3_27251118
identity

identity_1

identity_2

identity_3ѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallЖ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_27251077dense_27251079*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_272509132
dense/StatefulPartitionedCall╬
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*8
f3R1
/__inference_dense_activity_regularizer_272508682+
)dense/ActivityRegularizer/PartitionedCallў
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shapeе
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackг
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1г
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2■
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceф
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast╩
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivЊ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_27251090dense_1_27251092*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_272509602!
dense_1/StatefulPartitionedCallо
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*:
f5R3
1__inference_dense_1_activity_regularizer_272508832-
+dense_1/ActivityRegularizer/PartitionedCallъ
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shapeг
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack░
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1░
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2і
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice░
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Castм
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivЋ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_27251103dense_2_27251105*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_272510072!
dense_2/StatefulPartitionedCallо
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*:
f5R3
1__inference_dense_2_activity_regularizer_272508982-
+dense_2/ActivityRegularizer/PartitionedCallъ
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shapeг
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack░
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1░
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2і
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice░
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Castм
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truedivќ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_27251116dense_3_27251118*
Tin
2*
Tout
2*(
_output_shapes
:         ж*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_272510542!
dense_3/StatefulPartitionedCallЃ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*(
_output_shapes
:         ж2

IdentityЫ

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1З

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2З

Identity_3Identity'dense_2/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*G
_input_shapes6
4:         ж::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Q M
(
_output_shapes
:         ж
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┌
K
1__inference_dense_1_activity_regularizer_27250883
self
identity:
AbsAbsself*
T0*
_output_shapes
:2
Abs>
RankRankAbs:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeK
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/xM
addAddV2add/x:output:0mul:z:0*
T0*
_output_shapes
: 2
addJ
IdentityIdentityadd:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
┌
K
1__inference_dense_2_activity_regularizer_27250898
self
identity:
AbsAbsself*
T0*
_output_shapes
:2
Abs>
RankRankAbs:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeK
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/xM
addAddV2add/x:output:0mul:z:0*
T0*
_output_shapes
: 2
addJ
IdentityIdentityadd:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
Т
Г
E__inference_dense_1_layer_call_and_return_conditional_losses_27251549

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
сe
М
#__inference__wrapped_model_27250853
input_1.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource0
,model_dense_3_matmul_readvariableop_resource1
-model_dense_3_biasadd_readvariableop_resource
identityѕ▓
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	ж@*
dtype02#
!model/dense/MatMul/ReadVariableOpў
model/dense/MatMulMatMulinput_1)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/dense/MatMul░
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/dense/BiasAdd/ReadVariableOp▒
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
model/dense/ReluБ
#model/dense/ActivityRegularizer/AbsAbsmodel/dense/Relu:activations:0*
T0*'
_output_shapes
:         @2%
#model/dense/ActivityRegularizer/AbsЪ
%model/dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model/dense/ActivityRegularizer/Const╦
#model/dense/ActivityRegularizer/SumSum'model/dense/ActivityRegularizer/Abs:y:0.model/dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2%
#model/dense/ActivityRegularizer/SumЊ
%model/dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82'
%model/dense/ActivityRegularizer/mul/xл
#model/dense/ActivityRegularizer/mulMul.model/dense/ActivityRegularizer/mul/x:output:0,model/dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#model/dense/ActivityRegularizer/mulЊ
%model/dense/ActivityRegularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%model/dense/ActivityRegularizer/add/x═
#model/dense/ActivityRegularizer/addAddV2.model/dense/ActivityRegularizer/add/x:output:0'model/dense/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2%
#model/dense/ActivityRegularizer/addю
%model/dense/ActivityRegularizer/ShapeShapemodel/dense/Relu:activations:0*
T0*
_output_shapes
:2'
%model/dense/ActivityRegularizer/Shape┤
3model/dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3model/dense/ActivityRegularizer/strided_slice/stackИ
5model/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5model/dense/ActivityRegularizer/strided_slice/stack_1И
5model/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5model/dense/ActivityRegularizer/strided_slice/stack_2б
-model/dense/ActivityRegularizer/strided_sliceStridedSlice.model/dense/ActivityRegularizer/Shape:output:0<model/dense/ActivityRegularizer/strided_slice/stack:output:0>model/dense/ActivityRegularizer/strided_slice/stack_1:output:0>model/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-model/dense/ActivityRegularizer/strided_slice╝
$model/dense/ActivityRegularizer/CastCast6model/dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$model/dense/ActivityRegularizer/CastЛ
'model/dense/ActivityRegularizer/truedivRealDiv'model/dense/ActivityRegularizer/add:z:0(model/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'model/dense/ActivityRegularizer/truedivи
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02%
#model/dense_1/MatMul/ReadVariableOpх
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense_1/MatMulХ
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp╣
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense_1/BiasAddѓ
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model/dense_1/ReluЕ
%model/dense_1/ActivityRegularizer/AbsAbs model/dense_1/Relu:activations:0*
T0*'
_output_shapes
:          2'
%model/dense_1/ActivityRegularizer/AbsБ
'model/dense_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model/dense_1/ActivityRegularizer/ConstМ
%model/dense_1/ActivityRegularizer/SumSum)model/dense_1/ActivityRegularizer/Abs:y:00model/dense_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2'
%model/dense_1/ActivityRegularizer/SumЌ
'model/dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82)
'model/dense_1/ActivityRegularizer/mul/xп
%model/dense_1/ActivityRegularizer/mulMul0model/dense_1/ActivityRegularizer/mul/x:output:0.model/dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%model/dense_1/ActivityRegularizer/mulЌ
'model/dense_1/ActivityRegularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'model/dense_1/ActivityRegularizer/add/xН
%model/dense_1/ActivityRegularizer/addAddV20model/dense_1/ActivityRegularizer/add/x:output:0)model/dense_1/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2'
%model/dense_1/ActivityRegularizer/addб
'model/dense_1/ActivityRegularizer/ShapeShape model/dense_1/Relu:activations:0*
T0*
_output_shapes
:2)
'model/dense_1/ActivityRegularizer/ShapeИ
5model/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5model/dense_1/ActivityRegularizer/strided_slice/stack╝
7model/dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model/dense_1/ActivityRegularizer/strided_slice/stack_1╝
7model/dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model/dense_1/ActivityRegularizer/strided_slice/stack_2«
/model/dense_1/ActivityRegularizer/strided_sliceStridedSlice0model/dense_1/ActivityRegularizer/Shape:output:0>model/dense_1/ActivityRegularizer/strided_slice/stack:output:0@model/dense_1/ActivityRegularizer/strided_slice/stack_1:output:0@model/dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/model/dense_1/ActivityRegularizer/strided_slice┬
&model/dense_1/ActivityRegularizer/CastCast8model/dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2(
&model/dense_1/ActivityRegularizer/Cast┘
)model/dense_1/ActivityRegularizer/truedivRealDiv)model/dense_1/ActivityRegularizer/add:z:0*model/dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2+
)model/dense_1/ActivityRegularizer/truedivи
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02%
#model/dense_2/MatMul/ReadVariableOpи
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/dense_2/MatMulХ
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp╣
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/dense_2/BiasAddѓ
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
model/dense_2/ReluЕ
%model/dense_2/ActivityRegularizer/AbsAbs model/dense_2/Relu:activations:0*
T0*'
_output_shapes
:         @2'
%model/dense_2/ActivityRegularizer/AbsБ
'model/dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model/dense_2/ActivityRegularizer/ConstМ
%model/dense_2/ActivityRegularizer/SumSum)model/dense_2/ActivityRegularizer/Abs:y:00model/dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2'
%model/dense_2/ActivityRegularizer/SumЌ
'model/dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82)
'model/dense_2/ActivityRegularizer/mul/xп
%model/dense_2/ActivityRegularizer/mulMul0model/dense_2/ActivityRegularizer/mul/x:output:0.model/dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%model/dense_2/ActivityRegularizer/mulЌ
'model/dense_2/ActivityRegularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'model/dense_2/ActivityRegularizer/add/xН
%model/dense_2/ActivityRegularizer/addAddV20model/dense_2/ActivityRegularizer/add/x:output:0)model/dense_2/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2'
%model/dense_2/ActivityRegularizer/addб
'model/dense_2/ActivityRegularizer/ShapeShape model/dense_2/Relu:activations:0*
T0*
_output_shapes
:2)
'model/dense_2/ActivityRegularizer/ShapeИ
5model/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5model/dense_2/ActivityRegularizer/strided_slice/stack╝
7model/dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model/dense_2/ActivityRegularizer/strided_slice/stack_1╝
7model/dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model/dense_2/ActivityRegularizer/strided_slice/stack_2«
/model/dense_2/ActivityRegularizer/strided_sliceStridedSlice0model/dense_2/ActivityRegularizer/Shape:output:0>model/dense_2/ActivityRegularizer/strided_slice/stack:output:0@model/dense_2/ActivityRegularizer/strided_slice/stack_1:output:0@model/dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/model/dense_2/ActivityRegularizer/strided_slice┬
&model/dense_2/ActivityRegularizer/CastCast8model/dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2(
&model/dense_2/ActivityRegularizer/Cast┘
)model/dense_2/ActivityRegularizer/truedivRealDiv)model/dense_2/ActivityRegularizer/add:z:0*model/dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2+
)model/dense_2/ActivityRegularizer/truedivИ
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	@ж*
dtype02%
#model/dense_3/MatMul/ReadVariableOpИ
model/dense_3/MatMulMatMul model/dense_2/Relu:activations:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ж2
model/dense_3/MatMulи
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:ж*
dtype02&
$model/dense_3/BiasAdd/ReadVariableOp║
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ж2
model/dense_3/BiasAddї
model/dense_3/SigmoidSigmoidmodel/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         ж2
model/dense_3/Sigmoidn
IdentityIdentitymodel/dense_3/Sigmoid:y:0*
T0*(
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ж:::::::::Q M
(
_output_shapes
:         ж
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ќ`
Ы
C__inference_model_layer_call_and_return_conditional_losses_27251382

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3ѕа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ж@*
dtype02
dense/MatMul/ReadVariableOpЁ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2

dense/ReluЉ
dense/ActivityRegularizer/AbsAbsdense/Relu:activations:0*
T0*'
_output_shapes
:         @2
dense/ActivityRegularizer/AbsЊ
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
dense/ActivityRegularizer/Const│
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/SumЄ
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82!
dense/ActivityRegularizer/mul/xИ
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/mulЄ
dense/ActivityRegularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense/ActivityRegularizer/add/xх
dense/ActivityRegularizer/addAddV2(dense/ActivityRegularizer/add/x:output:0!dense/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/addі
dense/ActivityRegularizer/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shapeе
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackг
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1г
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2■
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceф
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast╣
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/add:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivЦ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_1/ReluЌ
dense_1/ActivityRegularizer/AbsAbsdense_1/Relu:activations:0*
T0*'
_output_shapes
:          2!
dense_1/ActivityRegularizer/AbsЌ
!dense_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_1/ActivityRegularizer/Const╗
dense_1/ActivityRegularizer/SumSum#dense_1/ActivityRegularizer/Abs:y:0*dense_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/SumІ
!dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82#
!dense_1/ActivityRegularizer/mul/x└
dense_1/ActivityRegularizer/mulMul*dense_1/ActivityRegularizer/mul/x:output:0(dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/mulІ
!dense_1/ActivityRegularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_1/ActivityRegularizer/add/xй
dense_1/ActivityRegularizer/addAddV2*dense_1/ActivityRegularizer/add/x:output:0#dense_1/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/addљ
!dense_1/ActivityRegularizer/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shapeг
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack░
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1░
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2і
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice░
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Cast┴
#dense_1/ActivityRegularizer/truedivRealDiv#dense_1/ActivityRegularizer/add:z:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivЦ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02
dense_2/MatMul/ReadVariableOpЪ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_2/MatMulц
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOpА
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
dense_2/ReluЌ
dense_2/ActivityRegularizer/AbsAbsdense_2/Relu:activations:0*
T0*'
_output_shapes
:         @2!
dense_2/ActivityRegularizer/AbsЌ
!dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_2/ActivityRegularizer/Const╗
dense_2/ActivityRegularizer/SumSum#dense_2/ActivityRegularizer/Abs:y:0*dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/SumІ
!dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82#
!dense_2/ActivityRegularizer/mul/x└
dense_2/ActivityRegularizer/mulMul*dense_2/ActivityRegularizer/mul/x:output:0(dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/mulІ
!dense_2/ActivityRegularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_2/ActivityRegularizer/add/xй
dense_2/ActivityRegularizer/addAddV2*dense_2/ActivityRegularizer/add/x:output:0#dense_2/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/addљ
!dense_2/ActivityRegularizer/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shapeг
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack░
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1░
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2і
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice░
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast┴
#dense_2/ActivityRegularizer/truedivRealDiv#dense_2/ActivityRegularizer/add:z:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truedivд
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@ж*
dtype02
dense_3/MatMul/ReadVariableOpа
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ж2
dense_3/MatMulЦ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:ж*
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ж2
dense_3/BiasAddz
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         ж2
dense_3/Sigmoidh
IdentityIdentitydense_3/Sigmoid:y:0*
T0*(
_output_shapes
:         ж2

Identityl

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2

Identity_1n

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2

Identity_2n

Identity_3Identity'dense_2/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*G
_input_shapes6
4:         ж:::::::::P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ь
Г
E__inference_dense_3_layer_call_and_return_conditional_losses_27251054

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@ж*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ж2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ж*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ж2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         ж2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
йH
┼
C__inference_model_layer_call_and_return_conditional_losses_27251074
input_1
dense_27250936
dense_27250938
dense_1_27250983
dense_1_27250985
dense_2_27251030
dense_2_27251032
dense_3_27251065
dense_3_27251067
identity

identity_1

identity_2

identity_3ѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallЖ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_27250936dense_27250938*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_272509132
dense/StatefulPartitionedCall╬
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*8
f3R1
/__inference_dense_activity_regularizer_272508682+
)dense/ActivityRegularizer/PartitionedCallў
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shapeе
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackг
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1г
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2■
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceф
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast╩
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivЊ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_27250983dense_1_27250985*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_272509602!
dense_1/StatefulPartitionedCallо
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*:
f5R3
1__inference_dense_1_activity_regularizer_272508832-
+dense_1/ActivityRegularizer/PartitionedCallъ
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shapeг
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack░
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1░
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2і
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice░
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Castм
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivЋ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_27251030dense_2_27251032*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_272510072!
dense_2/StatefulPartitionedCallо
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*:
f5R3
1__inference_dense_2_activity_regularizer_272508982-
+dense_2/ActivityRegularizer/PartitionedCallъ
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shapeг
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack░
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1░
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2і
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice░
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Castм
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truedivќ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_27251065dense_3_27251067*
Tin
2*
Tout
2*(
_output_shapes
:         ж*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_272510542!
dense_3/StatefulPartitionedCallЃ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*(
_output_shapes
:         ж2

IdentityЫ

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1З

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2З

Identity_3Identity'dense_2/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*G
_input_shapes6
4:         ж::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Q M
(
_output_shapes
:         ж
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Э

*__inference_dense_2_layer_call_fn_27251589

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_272510072
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╬	
о
&__inference_signature_wrapper_27251305
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*(
_output_shapes
:         ж**
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__wrapped_model_272508532
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ж
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
█

г
G__inference_dense_layer_call_and_return_all_conditional_losses_27251538

inputs
unknown
	unknown_0
identity

identity_1ѕбStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_272509132
StatefulPartitionedCallћ
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*8
f3R1
/__inference_dense_activity_regularizer_272508682
PartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*/
_input_shapes
:         ж::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ь
Г
E__inference_dense_3_layer_call_and_return_conditional_losses_27251611

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@ж*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ж2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ж*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ж2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         ж2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
п
I
/__inference_dense_activity_regularizer_27250868
self
identity:
AbsAbsself*
T0*
_output_shapes
:2
Abs>
RankRankAbs:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         2
rangeK
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *иЛ82
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulS
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/xM
addAddV2add/x:output:0mul:z:0*
T0*
_output_shapes
: 2
addJ
IdentityIdentityadd:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
щ	
п
(__inference_model_layer_call_fn_27251201
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*.
_output_shapes
:         ж: : : **
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_272511792
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ж
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "»L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*г
serving_defaultў
<
input_11
serving_default_input_1:0         ж<
dense_31
StatefulPartitionedCall:0         жtensorflow/serving/predict:╔╝
б/
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
V__call__
W_default_save_signature
*X&call_and_return_all_conditional_losses"«,
_tf_keras_modelћ,{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 745]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 745, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 745]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 745]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 745, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_3", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adadelta", "config": {"name": "Adadelta", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.949999988079071, "epsilon": 1e-07}}}}
ь"Ж
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 745]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 745]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ш

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"л
_tf_keras_layerХ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 745}}}, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 745]}}
э

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"м
_tf_keras_layerИ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
э

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"м
_tf_keras_layerИ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Л

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
___call__
*`&call_and_return_all_conditional_losses"г
_tf_keras_layerњ{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 745, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
▄
$iter
	%decay
&learning_rate
'rho
accum_gradF
accum_gradG
accum_gradH
accum_gradI
accum_gradJ
accum_gradK
accum_gradL
accum_gradM	accum_varN	accum_varO	accum_varP	accum_varQ	accum_varR	accum_varS	accum_varT	accum_varU"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
╩
(layer_regularization_losses
	variables
regularization_losses

)layers
*metrics
	trainable_variables
+non_trainable_variables
,layer_metrics
V__call__
W_default_save_signature
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
:	ж@2dense/kernel
:@2
dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
╩
-layer_regularization_losses
	variables
regularization_losses

.layers
/metrics
trainable_variables
0non_trainable_variables
1layer_metrics
Y__call__
bactivity_regularizer_fn
*Z&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 :@ 2dense_1/kernel
: 2dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
╩
2layer_regularization_losses
	variables
regularization_losses

3layers
4metrics
trainable_variables
5non_trainable_variables
6layer_metrics
[__call__
dactivity_regularizer_fn
*\&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 : @2dense_2/kernel
:@2dense_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
╩
7layer_regularization_losses
	variables
regularization_losses

8layers
9metrics
trainable_variables
:non_trainable_variables
;layer_metrics
]__call__
factivity_regularizer_fn
*^&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
!:	@ж2dense_3/kernel
:ж2dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
<layer_regularization_losses
 	variables
!regularization_losses

=layers
>metrics
"trainable_variables
?non_trainable_variables
@layer_metrics
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
A0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╗
	Btotal
	Ccount
D	variables
E	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
B0
C1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
1:/	ж@2 Adadelta/dense/kernel/accum_grad
*:(@2Adadelta/dense/bias/accum_grad
2:0@ 2"Adadelta/dense_1/kernel/accum_grad
,:* 2 Adadelta/dense_1/bias/accum_grad
2:0 @2"Adadelta/dense_2/kernel/accum_grad
,:*@2 Adadelta/dense_2/bias/accum_grad
3:1	@ж2"Adadelta/dense_3/kernel/accum_grad
-:+ж2 Adadelta/dense_3/bias/accum_grad
0:.	ж@2Adadelta/dense/kernel/accum_var
):'@2Adadelta/dense/bias/accum_var
1:/@ 2!Adadelta/dense_1/kernel/accum_var
+:) 2Adadelta/dense_1/bias/accum_var
1:/ @2!Adadelta/dense_2/kernel/accum_var
+:)@2Adadelta/dense_2/bias/accum_var
2:0	@ж2!Adadelta/dense_3/kernel/accum_var
,:*ж2Adadelta/dense_3/bias/accum_var
Ь2в
(__inference_model_layer_call_fn_27251483
(__inference_model_layer_call_fn_27251201
(__inference_model_layer_call_fn_27251507
(__inference_model_layer_call_fn_27251276└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Р2▀
#__inference__wrapped_model_27250853и
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *'б$
"і
input_1         ж
┌2О
C__inference_model_layer_call_and_return_conditional_losses_27251382
C__inference_model_layer_call_and_return_conditional_losses_27251459
C__inference_model_layer_call_and_return_conditional_losses_27251125
C__inference_model_layer_call_and_return_conditional_losses_27251074└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
(__inference_dense_layer_call_fn_27251527б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ы2Ь
G__inference_dense_layer_call_and_return_all_conditional_losses_27251538б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_dense_1_layer_call_fn_27251558б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_dense_1_layer_call_and_return_all_conditional_losses_27251569б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_dense_2_layer_call_fn_27251589б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_dense_2_layer_call_and_return_all_conditional_losses_27251600б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_dense_3_layer_call_fn_27251620б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_dense_3_layer_call_and_return_conditional_losses_27251611б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5B3
&__inference_signature_wrapper_27251305input_1
я2█
/__inference_dense_activity_regularizer_27250868Д
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
ь2Ж
C__inference_dense_layer_call_and_return_conditional_losses_27251518б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Я2П
1__inference_dense_1_activity_regularizer_27250883Д
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
№2В
E__inference_dense_1_layer_call_and_return_conditional_losses_27251549б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Я2П
1__inference_dense_2_activity_regularizer_27250898Д
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
№2В
E__inference_dense_2_layer_call_and_return_conditional_losses_27251580б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ў
#__inference__wrapped_model_27250853q1б.
'б$
"і
input_1         ж
ф "2ф/
-
dense_3"і
dense_3         ж^
1__inference_dense_1_activity_regularizer_27250883)б
б
і
self
ф "і и
I__inference_dense_1_layer_call_and_return_all_conditional_losses_27251569j/б,
%б"
 і
inputs         @
ф "3б0
і
0          
џ
і	
1/0 Ц
E__inference_dense_1_layer_call_and_return_conditional_losses_27251549\/б,
%б"
 і
inputs         @
ф "%б"
і
0          
џ }
*__inference_dense_1_layer_call_fn_27251558O/б,
%б"
 і
inputs         @
ф "і          ^
1__inference_dense_2_activity_regularizer_27250898)б
б
і
self
ф "і и
I__inference_dense_2_layer_call_and_return_all_conditional_losses_27251600j/б,
%б"
 і
inputs          
ф "3б0
і
0         @
џ
і	
1/0 Ц
E__inference_dense_2_layer_call_and_return_conditional_losses_27251580\/б,
%б"
 і
inputs          
ф "%б"
і
0         @
џ }
*__inference_dense_2_layer_call_fn_27251589O/б,
%б"
 і
inputs          
ф "і         @д
E__inference_dense_3_layer_call_and_return_conditional_losses_27251611]/б,
%б"
 і
inputs         @
ф "&б#
і
0         ж
џ ~
*__inference_dense_3_layer_call_fn_27251620P/б,
%б"
 і
inputs         @
ф "і         ж\
/__inference_dense_activity_regularizer_27250868)б
б
і
self
ф "і Х
G__inference_dense_layer_call_and_return_all_conditional_losses_27251538k0б-
&б#
!і
inputs         ж
ф "3б0
і
0         @
џ
і	
1/0 ц
C__inference_dense_layer_call_and_return_conditional_losses_27251518]0б-
&б#
!і
inputs         ж
ф "%б"
і
0         @
џ |
(__inference_dense_layer_call_fn_27251527P0б-
&б#
!і
inputs         ж
ф "і         @▀
C__inference_model_layer_call_and_return_conditional_losses_27251074Ќ9б6
/б,
"і
input_1         ж
p

 
ф "PбM
і
0         ж
-џ*
і	
1/0 
і	
1/1 
і	
1/2 ▀
C__inference_model_layer_call_and_return_conditional_losses_27251125Ќ9б6
/б,
"і
input_1         ж
p 

 
ф "PбM
і
0         ж
-џ*
і	
1/0 
і	
1/1 
і	
1/2 я
C__inference_model_layer_call_and_return_conditional_losses_27251382ќ8б5
.б+
!і
inputs         ж
p

 
ф "PбM
і
0         ж
-џ*
і	
1/0 
і	
1/1 
і	
1/2 я
C__inference_model_layer_call_and_return_conditional_losses_27251459ќ8б5
.б+
!і
inputs         ж
p 

 
ф "PбM
і
0         ж
-џ*
і	
1/0 
і	
1/1 
і	
1/2 ї
(__inference_model_layer_call_fn_27251201`9б6
/б,
"і
input_1         ж
p

 
ф "і         жї
(__inference_model_layer_call_fn_27251276`9б6
/б,
"і
input_1         ж
p 

 
ф "і         жІ
(__inference_model_layer_call_fn_27251483_8б5
.б+
!і
inputs         ж
p

 
ф "і         жІ
(__inference_model_layer_call_fn_27251507_8б5
.б+
!і
inputs         ж
p 

 
ф "і         жд
&__inference_signature_wrapper_27251305|<б9
б 
2ф/
-
input_1"і
input_1         ж"2ф/
-
dense_3"і
dense_3         ж