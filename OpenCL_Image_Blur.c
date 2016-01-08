#include <stdio.h>
#include <stdlib.h>

#include "pgm.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#define MAX_SOURCE_SIZE (0x100000) //1MB

#define DEBUG 1


/*
int main ( int arc, char **argv )
{

s


}
*/



void printDebug(char string[])
{
	if (DEBUG == 1) fprintf(stdout, "DEBUG: %s %s",string,"\n");
}

void printError(char string[])
{
	fprintf(stderr, "ERROR: %s %s",string,"\n");
}


/**
 * [IN] fileName = the name of the kernel file
 * [OUT] source_str = the file content
 * [OUT] source_size = the file size
 */
void readKernel(char fileName[],char* source_str,size_t* source_size)
{
	  FILE *fp;

	  fp = fopen(fileName, "r");
	  if (fp == NULL)
	  {
		  fprintf(stderr, "ERROR: Failed to load kernel.\n");
		  exit(1);
	  }

	  *source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	  fclose( fp );
}



int main()
{
  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;

  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  printDebug("Program started");

//REGION load the kernel file
  char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
  size_t source_size;
  readKernel("./OpenCL_Image_Blur_Kernel.cl",source_str,&source_size);

  printDebug("Kernel Loaded");
//ENDREGION

//REGION load the image
  unsigned char* img = NULL;
  int width;
  int height;
  int status = pgm_load(&img,&height,&width,"images\\Lena-512x512.pgm");
  if (status == 0)
  {
	  printf("Image Loaded %dx%d \n",width,height);
  }
  else
  {
	  printError("Image can't be loaded");
	  exit(1);
  }
//ENDREGION



  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  if (ret == CL_SUCCESS)
  {
  	  printDebug("clGetPlatformIDs OK");
  }
  else
  {
	  printError("clGetPlatformIDs ERROR");
   	   exit(1);
  }
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
  if (ret == CL_SUCCESS)
  {
  	  printDebug("clGetDeviceIDs OK");
  }
  else
  {
	  printError("clGetDeviceIDs ERROR");
   	   exit(1);
  }
  context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  if (ret == CL_SUCCESS)
  {
  	  printDebug("clCreateContext OK");
  }
  else
  {
	  printError("clCreateContext ERROR");
   	   exit(1);
  }
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  if (ret == CL_SUCCESS)
  {
  	  printDebug("command_queue OK");
  }
  else
  {
	  printError("command_queue ERROR");
   	   exit(1);
  }

  cl_mem inputImage = clCreateBuffer(context, CL_MEM_READ_ONLY,width*height * sizeof(unsigned char), NULL, &ret);
  ret = clEnqueueWriteBuffer(command_queue,inputImage,CL_TRUE,0,width*height,img,NULL,NULL,NULL);
  if (ret == CL_SUCCESS)
  {
	  printDebug("clCreateBuffer inputImage OK");
  }
  else
  {
	  printError("clCreateBuffer inputImage");
	  exit(1);
  }
  cl_mem outputImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY,width*height * sizeof(unsigned char), NULL, &ret);
  if (ret == CL_SUCCESS)
  {
	  printDebug("clCreateBuffer outputImage OK");
  }
  else
  {
	  printError("clCreateBuffer outputImage");
	  exit(1);
  }
//REGION build kernel program
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,(const size_t *)&source_size, &ret);
  if (ret == CL_SUCCESS)
  {
	  printDebug("clCreateProgramWithSource OK");
  }
  else
  {
	  printError("clCreateProgramWithSource");
	  exit(1);
  }
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (ret == CL_SUCCESS)
  {
	  printDebug("clBuildProgram OK");
  }
  else
  {
	  printError("clBuildProgram");
	  exit(1);
  }
  kernel = clCreateKernel(program, "blur", &ret);
  if (ret == CL_SUCCESS)
  {
  	  printDebug("clCreateKernel OK");
  }
  else
  {
	  printError("clCreateKernel");
   	   exit(1);
  }
//ENDREGION

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputImage);
  if (ret == CL_SUCCESS)
  {
  	  printDebug("clSetKernelArg 0 OK");
  }
  else
  {
	  printError("clSetKernelArg 0 PROBLEM");
   	   exit(1);
  }
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inputImage);
  if (ret == CL_SUCCESS)
  {
  	  printDebug("clSetKernelArg 1 OK");
  }
  else
  {
	  printError("clSetKernelArg 1 PROBLEM");
   	   exit(1);
  }
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputImage);
  if (ret == CL_SUCCESS)
  {
  	  printDebug("clSetKernelArg 2 OK");
  }
  else
  {
	  printError("clSetKernelArg 2 ERROR");
   	   exit(1);
  }

  const size_t  global_size = 512*512;
  const size_t  local_size = 1;
  ret = clEnqueueNDRangeKernel(	command_queue,kernel,2,NULL,&global_size,&local_size,0,NULL,NULL);
  //ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
  if (ret == CL_SUCCESS)
  {
  	  printDebug("clEnqueueNDRangeKernel OK");
  }
  else
  {
	  printError("clEnqueueNDRangeKernel ERROR");
   	   exit(1);
  }

  unsigned char* imgOutput = malloc(width*height*sizeof(unsigned char));
  if (imgOutput == NULL)
  {
	  printError("Can't allocate imgOutput buffer");
	  exit(1);
  }
  ret = clEnqueueReadBuffer(command_queue, outputImage, CL_TRUE, 0,width*height  * sizeof(char),imgOutput, 0, NULL, NULL);
  if (ret == CL_SUCCESS)
  {
  	  printDebug("clEnqueueReadBuffer OK");
  }
  else
  {
	  printError("clEnqueueReadBuffer ERROR");
   	   exit(1);
  }

  //REGION save the image
      status = pgm_save(imgOutput,height,width,"images_output\\Lena-512x512_BLURRED.pgm");
      if (status == 0)
      {
    	  printDebug("Image Saved");
      }
      else
      {
    	  printError("Image can't be saved");
    	  exit(1);
      }
  //ENDREGION


  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(inputImage);
  ret = clReleaseMemObject(outputImage);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  printDebug("Kernel File Freed from memory");
  free(source_str);
  free(imgOutput);

  return 0;
}

