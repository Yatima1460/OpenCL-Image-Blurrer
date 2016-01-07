#include <stdio.h>
#include <stdlib.h>



#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000) //1MB

#define DEBUG 1


/*
int main ( int arc, char **argv )
{




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
  cl_mem memobj = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  printDebug("Program started");

  char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
  size_t source_size;
  readKernel("./OpenCL_Image_Blur_Kernel.cl",source_str,&source_size);

  printDebug("Kernel Loaded");

  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

  memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,MEM_SIZE * sizeof(char), NULL, &ret);

  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
				                      (const size_t *)&source_size, &ret);


  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);


  kernel = clCreateKernel(program, "hello", &ret);


  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);


  ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);


  char string[MEM_SIZE];
  ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
			                MEM_SIZE * sizeof(char),string, 0, NULL, NULL);


  puts(string);


  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  printDebug("Kernel File Freed from memory");
  free(source_str);

  return 0;
}

