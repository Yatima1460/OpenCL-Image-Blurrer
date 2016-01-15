#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void blur(__global unsigned char* inputImage,__global unsigned char* blurFilter,__global unsigned char* outputImage,
                   int inputWidth, int inputHeight, int fliterSize)
{
  	 
    int indexX = get_global_id(0);
    int indexJ = get_global_id(1);
    if(indexX +  indexJ * inputWidth > inputWidth * inputHeight)
        return;
    
    int fIndex = 0;
    int sum = 0;
    int diff = 0;
    int diffY = indexJ - fliterSize / 2;
    for(fIndex = 0; fIndex < fliterSize; fIndex++)
    {
        if(fIndex > fliterSize / 2)
            diff += 2;
        int beginX = IndexX - fIndex;
        int endX = IndexX + fIndex + 1;
        diffY++;
        int i;
        for(i = begin; i < end; i++)
            sum += inputImage[i + diffY * inputWidth};
    }
    
    int normalization = 0;
    
    for(int i = 1; i <= fliterSize; i++)
        normalization = normalization * 2 + i;
    
    outputImage[indexX + inputWidth * indexJ] = sum / normalization;
}
