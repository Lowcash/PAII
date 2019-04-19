#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <random>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <glew.h>
#include <freeglut.h>
#include <cudaDefs.h>
#include <imageManager.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <math_functions.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_gl.h>      // helper functions for CUDA/GL interop

#include "imageKernels.cuh"

#define BLOCK_DIM 8

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

//CUDA variables
unsigned int imageWidth;
unsigned int imageHeight;
unsigned int imageBPP;		//Bits Per Pixel = 8, 16, 24, or 32 bit
unsigned int imagePitch;
cudaGraphicsResource_t cudaPBOResource;
cudaGraphicsResource_t cudaTexResource;
texture<uchar4, 2, cudaReadModeElementType> cudaTexRef;
cudaChannelFormatDesc cudaTexChannelDesc;
KernelSetting ks;

//OpenGL
unsigned int pboID;
unsigned int textureID;

unsigned int viewportWidth = 1024;
unsigned int viewportHeight = 1024;

//Application
constexpr unsigned int NUM_OF_RAIN_DROPS = 1 << 20;

bool isGradientCalculated = false;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937_64 generator(seed);

struct RainDrop {
private:

public:
	RainDrop()
		: gridPosition(int2()), exactPosition(double2()), directionForce(double2()), isDrop(false) {}

	void setPosition(const int posX, const int posY) {
		this->gridPosition = make_int2(posX, posY);
		this->exactPosition = make_double2(posX, posY);
	}
	void setDrop(const bool isDrop) {
		this->isDrop = isDrop;
	}

	bool isDropOnPosition() { return this->isDrop; }

	int2 gridPosition;
	double2 exactPosition;
	double2 directionForce;
	bool isDrop;
};

struct RainBuffer {
	RainBuffer()
		: actualDropState(0), promiseDropState(0), directionForce(int2()) {}

	void addDropsToState(const unsigned int drop) {
		actualDropState += drop;
		promiseDropState += drop;
	}

	int2 directionForce;

	float actualDropState;
	float promiseDropState;
};

RainDrop* hRainDropDevPtr = nullptr;
RainDrop* dRainDropDevPtr = nullptr;

RainBuffer* hRainBufferDevPtr = nullptr;
RainBuffer* dRainBufferDevPtr = nullptr;

size_t pitch;

void cudaWorker();
void loadTexture(const char* imageFileName);
void preparePBO();
void my_display();
void my_resize(GLsizei w, GLsizei h);
void my_idle();
void initGL(int argc, char **argv);
void releaseOpenGL();
void initCUDAtex();
void releaseCUDA();
void releaseApplication();
void releaseResources();
void assignRainBuffer();
void generateRandomDropsPosition(RainBuffer* rainBuffer, const unsigned int numOfDrops, const unsigned int width, const unsigned int height, const unsigned int depth);

__global__ void calculateGradients(const unsigned int pboWidth, const unsigned int pboHeight, unsigned char *pbo, RainBuffer* rainBuffer, const unsigned int pitch) {
	const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if(tx >= pboWidth || ty >= pboHeight) { return; }

	uchar4 tex = tex2D(cudaTexRef, tx, ty);

	const unsigned int rainBufferPitch = pitch / sizeof(RainBuffer);
	const unsigned int rainBufferOffset = ty * rainBufferPitch + tx;

	const unsigned int checkOffset = 1;

	const float lt = tex2D(cudaTexRef, tx - (1 * checkOffset), ty - (1 * checkOffset)).x;
	const float lc = tex2D(cudaTexRef, tx - (1 * checkOffset), ty + (0 * checkOffset)).x;
	const float lb = tex2D(cudaTexRef, tx - (1 * checkOffset), ty + (1 * checkOffset)).x;
	const float ct = tex2D(cudaTexRef, tx + (0 * checkOffset), ty - (1 * checkOffset)).x;
	const float cb = tex2D(cudaTexRef, tx + (0 * checkOffset), ty + (1 * checkOffset)).x;
	const float rt = tex2D(cudaTexRef, tx + (1 * checkOffset), ty - (1 * checkOffset)).x;
	const float rc = tex2D(cudaTexRef, tx + (1 * checkOffset), ty + (0 * checkOffset)).x;
	const float rb = tex2D(cudaTexRef, tx + (1 * checkOffset), ty + (1 * checkOffset)).x;

	int direction = 0;

	if(lt < tex.x) { direction = 1; tex.x = lt; }
	if(lc < tex.x) { direction = 2; tex.x = lc; }
	if(lb < tex.x) { direction = 3; tex.x = lb; }
	if(ct < tex.x) { direction = 4; tex.x = ct; }
	if(cb < tex.x) { direction = 5; tex.x = cb; }
	if(rt < tex.x) { direction = 6; tex.x = rt; }
	if(rc < tex.x) { direction = 7; tex.x = rc; }
	if(rb < tex.x) { direction = 8; tex.x = rb; }

	if(direction == 1) { rainBuffer[rainBufferOffset].directionForce = make_int2(-1, -1); }
	if(direction == 2) { rainBuffer[rainBufferOffset].directionForce = make_int2(-1,  0); }
	if(direction == 3) { rainBuffer[rainBufferOffset].directionForce = make_int2(-1,  1); }
	if(direction == 4) { rainBuffer[rainBufferOffset].directionForce = make_int2( 0, -1); }
	if(direction == 5) { rainBuffer[rainBufferOffset].directionForce = make_int2( 0,  1); }
	if(direction == 6) { rainBuffer[rainBufferOffset].directionForce = make_int2( 1, -1); }
	if(direction == 7) { rainBuffer[rainBufferOffset].directionForce = make_int2( 1,  0); }
	if(direction == 8) { rainBuffer[rainBufferOffset].directionForce = make_int2( 1,  1); }
}

__global__ void moveRainDrops(const unsigned int pboWidth, const unsigned int pboHeight, unsigned char *pbo, RainBuffer* rainBuffer, const unsigned int pitch) {
	const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if(tx >= pboWidth || ty >= pboHeight) { return; }

	const unsigned int rainBufferPitch = pitch / sizeof(RainBuffer);
	const unsigned int rainBufferOffset = ty * rainBufferPitch + tx;

	int2 directionForce = rainBuffer[rainBufferOffset].directionForce;

	const unsigned int flowTo = (ty + directionForce.y) * rainBufferPitch + (tx + directionForce.x);

	if(flowTo > 0 && flowTo < pboWidth * pboHeight) { 
		rainBuffer[flowTo].promiseDropState += rainBuffer[rainBufferOffset].actualDropState * 0.025;

		rainBuffer[rainBufferOffset].promiseDropState -= rainBuffer[rainBufferOffset].actualDropState * 0.025;

		rainBuffer[rainBufferOffset].promiseDropState = max(rainBuffer[rainBufferOffset].promiseDropState, 0.0);
	}
}

__global__ void vizualizeRainDrops(const unsigned int pboWidth, const unsigned int pboHeight, unsigned char *pbo, RainBuffer* rainBuffer, const unsigned int pitch) {
	const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if(tx >= pboWidth || ty >= pboHeight) { return; }

	uchar4 tex = tex2D(cudaTexRef, tx, ty);

	const unsigned int texOffset = (ty * pboWidth + tx) * 4;
	//pbo[texOffset + 0] = tex.x; pbo[texOffset + 1] = tex.y; pbo[texOffset + 2] = tex.z; pbo[texOffset + 3] = tex.w;
	pbo[texOffset + 0] = 0; pbo[texOffset + 1] = 0; pbo[texOffset + 2] = 0; pbo[texOffset + 3] = 0;

	const unsigned int rainBufferPitch = pitch / sizeof(RainBuffer);
	const unsigned int rainBufferOffset = ty * rainBufferPitch + tx;

	rainBuffer[rainBufferOffset].actualDropState = rainBuffer[rainBufferOffset].promiseDropState;

	//pbo[texOffset + 2] = max((int)rainBuffer[rainBufferOffset].actualDropState, pbo[texOffset + 2]);
	if(rainBuffer[rainBufferOffset].actualDropState > 0) { pbo[texOffset + 2] = rainBuffer[rainBufferOffset].actualDropState; }
}

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	initGL(argc, argv);

	loadTexture("heightmap.png");
	//loadTexture("gradient.gif");

	preparePBO();

	initCUDAtex();

	hRainBufferDevPtr = new RainBuffer[imageWidth * imageHeight];

	generateRandomDropsPosition(hRainBufferDevPtr, NUM_OF_RAIN_DROPS, imageWidth, imageHeight, 1);

	assignRainBuffer();

	//start rendering mainloop
	glutMainLoop();
	atexit(releaseResources);
}

void assignRainBuffer() {
	checkCudaErrors(cudaMallocPitch((void**)&dRainBufferDevPtr, &pitch, imageHeight * sizeof(RainBuffer), imageWidth));
	checkCudaErrors(cudaMemcpy2D(dRainBufferDevPtr, pitch, hRainBufferDevPtr, imageHeight * sizeof(RainBuffer), imageHeight * sizeof(RainBuffer), imageWidth, cudaMemcpyHostToDevice));
}

void generateRandomDropsPosition(RainBuffer* rainBuffer, const unsigned int numOfDrops, const unsigned int width, const unsigned int height, const unsigned int depth) {
	std::uniform_int_distribution<int> dis_x(0, width);
	std::uniform_int_distribution<int> dis_y(0, height);

	for(unsigned int i = 0; i < numOfDrops; i++) {
		int index = (dis_y(generator) * width) + dis_x(generator);

		if(index < width * height)
			rainBuffer[index].addDropsToState(30);
	}
}

void cudaWorker()
{
	cudaArray* array;

	//TODO 3: Map cudaTexResource
	cudaGraphicsMapResources(1, &cudaTexResource, 0);

	//TODO 4: Get Mapped Array of cudaTexResource
	cudaGraphicsSubResourceGetMappedArray(&array, cudaTexResource, 0, 0);

	//TODO 5: Get cudaTexChannelDesc from previously obtained array
	cudaGetChannelDesc(&cudaTexChannelDesc, array);

	//TODO 6: Binf cudaTexRef to array
	cudaBindTextureToArray(&cudaTexRef, array, &cudaTexChannelDesc);
	checkError();

	unsigned char *pboData;
	size_t pboSize;
	//TODO 7: Map cudaPBOResource
	cudaGraphicsMapResources(1, &cudaPBOResource, 0);

	//TODO 7: Map Mapped pointer to cudaPBOResource data
	cudaGraphicsResourceGetMappedPointer((void**)&pboData, &pboSize, cudaPBOResource);
	checkError();

	//TODO 8: Set KernelSetting variable ks (dimBlock, dimGrid, etc.) such that block will have BLOCK_DIM x BLOCK_DIM threads
	ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.dimGrid = dim3((imageWidth + BLOCK_DIM - 1) / BLOCK_DIM, (imageHeight + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	//Calling kernels
	if(!isGradientCalculated) {
		calculateGradients << <ks.dimGrid, ks.dimBlock >> > (imageWidth, imageHeight, pboData, dRainBufferDevPtr, pitch);

		isGradientCalculated = true;
	}

	moveRainDrops << <ks.dimGrid, ks.dimBlock >> > (imageWidth, imageHeight, pboData, dRainBufferDevPtr, pitch);
	vizualizeRainDrops << <ks.dimGrid, ks.dimBlock >> > (imageWidth, imageHeight, pboData, dRainBufferDevPtr, pitch);

	//Following code release mapped resources, unbinds texture and ensures that PBO data will be coppied into OpenGL texture. Do not modify following code!
	cudaUnbindTexture(&cudaTexRef);
	cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);
	cudaGraphicsUnmapResources(1, &cudaTexResource, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory
}

#pragma region OpenGL Routines - DO NOT MODIFY THIS SECTION !!!

void loadTexture(const char* imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);

	imageWidth = FreeImage_GetWidth(tmp);
	imageHeight = FreeImage_GetHeight(tmp);
	imageBPP = FreeImage_GetBPP(tmp);
	imagePitch = FreeImage_GetPitch(tmp);

	tmp = FreeImage_ConvertTo32Bits(tmp);

	//OpenGL Texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	//WARNING: Just some of inner format are supported by CUDA!!!
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, FreeImage_GetBits(tmp));
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	FreeImage_Unload(tmp);
}

void preparePBO()
{
	glGenBuffers(1, &pboID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);												// Make this the current UNPACK buffer (OpenGL is state-based)
	glBufferData(GL_PIXEL_UNPACK_BUFFER, imageWidth * imageHeight * 4, NULL, GL_DYNAMIC_COPY);	// Allocate data for the buffer. 4-channel 8-bit image
}

void my_display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, textureID);

	//I know this is a very old OpenGL, but we want to practice CUDA :-)
		//Now it will be a wasted time to learn you current features of OpenGL. Sorry for that however, you can visit my second seminar dealing with Computer Graphics (CG2).
	glBegin(GL_QUADS);

	glTexCoord2d(0, 0);		glVertex2d(0, 0);
	glTexCoord2d(1, 0);		glVertex2d(viewportWidth, 0);
	glTexCoord2d(1, 1);		glVertex2d(viewportWidth, viewportHeight);
	glTexCoord2d(0, 1);		glVertex2d(0, viewportHeight);

	glEnd();

	glDisable(GL_TEXTURE_2D);

	glFlush();
	glutSwapBuffers();
}

void my_resize(GLsizei w, GLsizei h)
{
	viewportWidth = w;
	viewportHeight = h;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, viewportWidth, viewportHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, viewportWidth, 0, viewportHeight);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
}

void my_idle()
{
	cudaWorker();
	glutPostRedisplay();
}

void initGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(viewportWidth, viewportHeight);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(":-)");

	glutDisplayFunc(my_display);
	glutReshapeFunc(my_resize);
	glutIdleFunc(my_idle);
	glutSetCursor(GLUT_CURSOR_CROSSHAIR);

	// initialize necessary OpenGL extensions
	glewInit();

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glShadeModel(GL_SMOOTH);
	glViewport(0, 0, viewportWidth, viewportHeight);

	glFlush();
}

void releaseOpenGL()
{
	if(textureID > 0)
		glDeleteTextures(1, &textureID);
	if(pboID > 0)
		glDeleteBuffers(1, &pboID);
}

#pragma endregion

#pragma region CUDA Routines

void initCUDAtex()
{
	cudaGLSetGLDevice(0);
	checkError();

	//CUDA Texture settings
	cudaTexRef.normalized = false;						//Otherwise TRUE to access with normalized texture coordinates
	cudaTexRef.filterMode = cudaFilterModePoint;		//Otherwise texRef.filterMode = cudaFilterModeLinear; for Linear interpolation of texels
	cudaTexRef.addressMode[0] = cudaAddressModeClamp;	//No repeat texture pattern
	cudaTexRef.addressMode[1] = cudaAddressModeClamp;	//No repeat texture pattern

	//TODO 1: Register OpenGL texture to CUDA resource
	cudaGraphicsGLRegisterImage(&cudaTexResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
	checkError();

	//TODO 2: Register PBO to CUDA resource
	cudaGraphicsGLRegisterBuffer(&cudaPBOResource, pboID, cudaGraphicsRegisterFlagsWriteDiscard);
	checkError();
}

void releaseCUDA()
{
	cudaGraphicsUnregisterResource(cudaPBOResource);
	cudaGraphicsUnregisterResource(cudaTexResource);
}

#pragma endregion

void releaseApplication() {
	delete[] hRainBufferDevPtr;
	delete[] dRainBufferDevPtr;
}

void releaseResources()
{
	releaseCUDA();
	releaseOpenGL();
	releaseApplication();
}
