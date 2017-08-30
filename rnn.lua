require 'image'
require 'nn'

--Following is function to traverse directory's folders
function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -a "'..directory..'"')
    for filename in pfile:lines() do
        i = i + 1
        t[i] = filename
    end
    pfile:close()
    return t
end

--Function to update state
function update_state(xk,sk,wx,wRec)
	local m=nn.HardTanh()
	local wx1=wx
	local wRec1=wRec
	wx1=torch.cmul(wx,xk)
	wRec1=torch.cmul(wRec,sk)
	return (wx1+wRec1)   
end

function forward_states(X,wx,wRec)
	S=torch.zeros(X:size(1),X:size(2)+1)
	local k
	for k=1,X:size(2)    --In for syntax for lua, both bounds are inclusive
	do
	S[{{},k+1}]=update_state(X[{{},k}],S[{{},k}],wx,wRec)
	end
	return S
end

function cost(y,t)
	return 2*(torch.pow((y-t),2))/nb_of_samples
end

function output_gradient(y,t)
	return 2*(y-t)/nb_of_samples --No need to declare nb_of_samples because variables in lua are global by default
end

function backward_gradient(X,S,grad_out,wRec)
	grad_over_time=torch.zeros(X:size(1),((X:size(2))+1))
	grad_over_time[{{},-1}]=grad_out
	wx_grad=0
	wRec_grad=0
	local m
	local k
	for m=X:size(2),1,-1
	do
	k=m+1
	wx_grad=wx_grad+grad_over_time[{{},k}]:cmul(X[{{},k-1}])
	wRec_grad=wRec_grad+grad_over_time[{{},k}]:cmul(S[{{},k-1}])
	grad_over_time[{{},k-1}]=grad_over_time[{{},k}]:cmul(wRec)	
	end
	return {wx_grad,wRec_grad},grad_over_time
end


function update_rprop(X,t,wx,wRec,W_prev_sign,W_delta,eta_p,eta_n)
	S=forward_states(X,wx,wRec)
	s=nn.SoftMax()
	grad_out=output_gradient(S[{{},-1}],t)
	W_grads,_=backward_gradient(X,S,grad_out,wRec)
	W_sign={}
	W_sign[1]=torch.sign(W_grads[1])
	W_sign[2]=torch.sign(W_grads[2])
	local i
	for i=1,wx:size(1)
	do
	if W_sign[1][i]==W_prev_sign[1][i] then
	W_delta[{{1},{i}}]=W_delta[{{1},{i}}]*eta_p
	else
	W_delta[{{1},{i}}]=W_delta[{{1},{i}}]*eta_n
	end	
	
	if W_sign[2][i]==W_prev_sign[2][i] then
	W_delta[{{2},{i}}]=W_delta[{{2},{i}}]*eta_p
	else
	W_delta[{{2},{i}}]=W_delta[{{2},{i}}]*eta_n
	end

	end
	return W_delta,W_sign 
end


--MAIN FUNCTION
	--The data for this code to work needs to be organized as follows: 
	--main_data_folder contains folders named 1, 2, 3 etc.
	--each folder in main_data_folder contains a sequence for the RNN. The images in each sequence are named 1.jpg, 2.jpg, 3.jpg, etc.
	image_list={}
	datasetname = '/path/to/main_data_folder'
	dirs=(scandir(datasetname))
	table.remove(dirs,1)
	table.remove(dirs,1)
	image_size=200
	leng=table.getn(dirs)
	print(leng)
	nb_of_samples=3 --This is the number of images in each sequence minus 1. Thus, 4-1=3
	sequence_len=200*200
	eta_p=1.25
	eta_n=0.5
	weight_size=sequence_len
	wx=2*torch.rand(1,weight_size)-1
	wRec=2*torch.rand(1,weight_size)-1
	
	W_delta=torch.ones(2,weight_size)
	W_delta=0.01*W_delta
	W_sign=torch.zeros(2,weight_size)

	local j
	local i
	local i1
	
	--Number of epochs can be defined using a loop here but this has been ignored for now to avoid complexity and compute power

	for j=1,leng-1
	do
		print(j)
		overallError=0
		image_list={}
		dir_name=datasetname.."/"..j.."/"  
		for pic_index=1,4  --4 should be replaced by the number of images in each sequence
		do
			filename=dir_name..pic_index..".JPG"
			print(filename)
			--IMAGE FUNCTIONS
			img=image.load(filename)
			img=image.scale(img,image_size,image_size)
			img1=torch.reshape(img,1,image_size*image_size)
			image_list[#image_list+1]=img1
			imgs_len=table.getn(image_list)
			X=torch.zeros(sequence_len,nb_of_samples)
		end
		for position=1,imgs_len-1
		do
			X[{{},position}]=image_list[position]	
		end		
		
		t=image_list[4]
		for i1=1,100 --number of iterations of training for one sequence
		do
			W_delta,W_sign=update_rprop(X,t,wx,wRec,W_sign,W_delta,eta_p,eta_n)
			for i=1,wx:size(1)
			do
				wx[i]=wx[i]-W_sign[1][i]*W_delta[1][i] 					
				wRec[i]=wRec[i]-W_sign[2][i]*W_delta[2][i]
			end
		
		end
	end


print(wRec[1][2])

--insert code here to save wx and wRec matrices 
--insert code here to load wx and wRec matrices

--TESTING
print("Testing")
input_len=3  --input_len is the number of images which will be given as input to test the model. Since the length of each sequence is 4, the input length is 4-1=3
xtest=torch.ones(sequence_len,input_len)

for i=1,input_len
do
title="/path/to/main_data_folder/1/"..i..".JPG"
img=image.load(title)
img=image.scale(img,image_size,image_size)
img1=torch.reshape(img,1,image_size*image_size)
xtest[{{},i}]=img1
end

test_outpt=forward_states(xtest,wx,wRec)
test_out=test_outpt[{{},-1}]
output=torch.reshape(test_out,image_size,image_size)

--Some machines face problems with the image.display() command. An error pops up saying "module 'qt' not found:No LuaRocks module found for qt". Fixing this may be quite difficult and thus, instead of trying to display the image, simply save it to a location and view the image once it has been saved. For more info see this - https://github.com/torch/image/issues/127
--image.display(output)
--print(output)
image.save("/path/to/save/image/result.png",output)
print("Output has been generated")


	


	

