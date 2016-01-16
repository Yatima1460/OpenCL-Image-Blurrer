#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void blur(__global unsigned char* inputImage,__global unsigned char* outputImage,
                   int inputWidth, int inputHeight, int filterSize)
{
  	int halfFilterSize = filterSize / 2; 
    int indexX = get_global_id(0);
    int indexY = get_global_id(1);
    if(indexX +  indexY * inputWidth > inputWidth * inputHeight)
        return;
    if(indexX - halfFilterSize < 0)
        return;
    if(indexY - halfFilterSize < 0)
        return;
    if(indexX + halfFilterSize > inputWidth)
        return;
    if(indexY + halfFilterSize > inputHeight)
        return;
    int fIndex = 0;
    int sum = 0;
    int diff = 0;
    int diffY = indexY - halfFilterSize;
    int k = 0;
    for(fIndex = 0; fIndex < filterSize; fIndex++)
    {
        if(fIndex > halfFilterSize)
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
    
    outputImage[indexX - halfFilterSize + inputWidth * (indexY - halfFilterSize)] = sum / k;
}
