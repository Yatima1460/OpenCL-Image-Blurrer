#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void blur(__global unsigned char* inputImage,__global unsigned char* blurFilter,__global unsigned char* outputImage)
{
    int x = get_global_id(0);
	int y = get_global_id(1);
    if (x < 300 && y < 300)
		outputImage[y*300+x] = ~inputImage[y*300+x];
}
