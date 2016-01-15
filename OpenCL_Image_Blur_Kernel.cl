#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void blur(__global unsigned char* inputImage,__global unsigned char* outputImage,int inputWidth, int inputHeight, int filterSize)
{
  	 
    int indexX = get_global_id(0);
    int indexY = get_global_id(1);
    if(indexX +  indexY * inputWidth > inputWidth * inputHeight)
        return;
    if(indexX - filterSize < 0)
        return;
    if(indexY - filterSize < 0)
        return;
    if(indexX + filterSize > inputWidth)
        return;
    if(indexY + filterSize > inputHeight)
        return;
    int fIndex = 0;
    int sum = 0;
    int diff = 0;
    int diffY = indexY - filterSize / 2;
    int k = 0;
    for(fIndex = 0; fIndex < filterSize; fIndex++)
    {
        if(fIndex > filterSize / 2)
            diff += 2;
        int beginX = indexX - fIndex;
        int endX = indexX + fIndex + 1;
        diffY++;
        int i;
        for(i = beginX; i < endX; i++)
        {
            sum += inputImage[i + diffY * inputWidth];
            k++;
        }
    }
    
    outputImage[indexX + inputWidth * indexY] = sum / k;
}
