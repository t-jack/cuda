#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

typedef int FILTER;

float filter[25] = {
    (float)1/256, (float)4/256, (float)6/256, (float)4/256, (float)1/256, 
    (float)4/256, (float)16/256, (float)24/256, (float)16/256, (float)4/256,
	(float)6/256, (float)24/256, (float)36/256, (float)24/256, (float)6/256,
    (float)4/256, (float)16/256, (float)64/256, (float)16/256, (float)4/256,
    (float)1/256, (float)4/256, (float)6/256, (float)4/256, (float)1/256
};

void filter_cpu(FILTER* pix_in, FILTER* pix_out, int x, int y)
{
	int i, j;
	int k, l;
    int sobel_x[9] = {
	    -1, 0, 1,
	    -2, 0, 2,
	    -1, 0, 1
	};
	FILTER temp;

    for(i=0;i<y;i++){
		for(j=0;j<x;j++){
			temp = 0;
			for(k=0;k<3;k++){
				for(l=0;l<3;l++){
					temp = temp + pix_in[j + l + (i + k)*(x+2)] * sobel_x[l + k*3];
				}
			}
			pix_out[j+1 + (i+1)*(x+2)] = temp;
		}
	}
}

__global__ void filter_gpu(FILTER* pix_in, FILTER* pix_out, int x, int y)
{
	int k, l;
	int thread_x = threadIdx.x % 32 + blockIdx.x * 32;
	int thread_y = threadIdx.x / 32 + blockIdx.y * 32;
    int sobel_x[9] = {
    	-1, 0, 1,
    	-2, 0, 2,
    	-1, 0, 1
    };
	FILTER temp = 0;

	for(k=0;k<3;k++){
		for(l=0;l<3;l++){
			temp = temp + pix_in[thread_x + l + (thread_y + k)*(x+2)] * sobel_x[l + k*3];
		}
	}
	pix_out[thread_x+1 + (thread_y+1)*(x+2)] = temp;
}

__global__ void filter_gpu_shared(FILTER* pix_in, FILTER* pix_out, int x, int y)
{
	int k, l;
	int thread_x = threadIdx.x % 32 + blockIdx.x * 32;
	int thread_y = threadIdx.x / 32 + blockIdx.y * 32;
    int sobel_x[9] = {
    	-1, 0, 1,
    	-2, 0, 2,
    	-1, 0, 1
	};
	int i, j;
	__shared__ FILTER pix_in_shared[34 * 34];
	FILTER temp = 0;

    for(i=0;i<34;i+=32){
		for(j=0;j<34;j+=32){
		    if((threadIdx.x%32) + j < 34 && (threadIdx.x/32) + i < 34){
				pix_in_shared[(threadIdx.x%32) + j + ((threadIdx.x/32) + i)*34] = pix_in[thread_x + j + (thread_y + i)*(x+2)];
			}
		}
	}

	__syncthreads();

	for(k=0;k<3;k++){
		for(l=0;l<3;l++){
			 temp = temp + pix_in_shared[(threadIdx.x%32) + l + ((threadIdx.x/32) + k)*34] * sobel_x[l + k*3];
		}
	}
	pix_out[thread_x+1 + (thread_y+1)*(x+2)] = temp;
}


int main(int argc, char *argv[])
{
	int i, j;
	FILE *fp;
	char buf[256];	//ヘッダ読み込みバッファ
	int type;
	
	int x, y, max, tmp;
	double ratio;
	int* pix;
	int bright[256];
	cudaEvent_t dev_start, dev_stop;
	float elapsedTime, h2dTime, d2hTime;

	FILTER *pix_in, *pix_out;

	cudaEventCreate(&dev_start);
	cudaEventCreate(&dev_stop);

	LARGE_INTEGER frequency, start, end;
	QueryPerformanceFrequency(&frequency);
	
	FILTER *pix_dev, *output;
	FILTER *pix_in_dev, *pix_out_dev;
	
	if((fp=fopen(argv[1], "r"))==NULL){
		printf("\nERROR : Cannot open file %s\n", argv[1]);
		exit(1);
	}
	
	/* -----ヘッダ取得----- */
	fgets(buf, 255, fp);
	if(buf[0]!='P')
		return -1;
	sscanf(buf, "P%d", &type);
	if(type!=2)
		return -1;
	
	do fgets(buf, 255, fp); while(buf[0]=='#'); /* コメント読み飛ばし */
	sscanf(buf, "%d%d", &x, &y); /* サイズ取得 */
	if(x<1 || y<1)
		return -1;
	
	
	do fgets(buf, 255, fp); while(buf[0]=='#'); /* コメント飛ばし */
	sscanf(buf, "%d", &max); /* 最大輝度の取得 */
	if(max<1 || max>255)
		return -1;
	ratio = 255/(float)max;
	
	/* -----データ取得----- */
	pix = (int*)malloc(sizeof(int)*x*y);
	output = (int*)malloc(sizeof(int)*x*y);
	pix_dev = (int*)malloc(sizeof(int)*x*y);
	for(i=0;i<y;i++){
		for(j=0;j<x;j++){
			if(fscanf(fp, "%d", &tmp)!=1) return -1;
			pix[j+i*x]=tmp*ratio;
		}
	}
    fclose(fp);
    
    printf("x : %d, y : %d\n", x, y);


    /* -----フィルタ用配列確保----- */
    pix_in = (FILTER*)malloc(sizeof(FILTER)*(x+2)*(y+2));
    pix_out = (FILTER*)malloc(sizeof(FILTER)*(x+2)*(y+2));
    for(i=0;i<y+2;i++){
        for(j=0;j<x+2;j++){
            pix_in[j + i*(x+2)] = 0;
        }
    }

    /* -----画像コピー----- */
    for(i=0;i<y;i++){
        for(j=0;j<x;j++){
            pix_in[j+1 + (i+1)*(x+2)] = (FILTER)pix[j + i*x];
        }
	}

	int l;
	QueryPerformanceCounter(&start);
	for(l=0;l<1;l++){
		filter_cpu(pix_in, pix_out, x, y);
	}
	QueryPerformanceCounter(&end);

    /* -----画像コピー----- */
    for(i=0;i<y;i++){
        for(j=0;j<x;j++){
			output[j + i*x] = (FILTER)pix_out[j+1 + (i+1)*(x+2)];
		}
	}

	cudaMalloc(&pix_in_dev, sizeof(FILTER)*(x+2)*(y+2));
	cudaMalloc(&pix_out_dev, sizeof(FILTER)*(x+2)*(y+2));

	cudaEventRecord(dev_start, 0);
	cudaMemcpy(pix_in_dev, pix_in, sizeof(FILTER)*(x+2)*(y+2), cudaMemcpyHostToDevice);
	cudaEventRecord(dev_stop, 0);
	cudaEventSynchronize(dev_stop);
	cudaEventElapsedTime(&h2dTime, dev_start, dev_stop);

	dim3 block (x/32, y/32);

	cudaEventRecord(dev_start, 0);
	filter_gpu_shared<<<block, 1024>>>(pix_in_dev, pix_out_dev, x, y);
	cudaEventRecord(dev_stop, 0);
	cudaEventSynchronize(dev_stop);
	cudaEventElapsedTime(&elapsedTime, dev_start, dev_stop);

	cudaEventRecord(dev_start, 0);
	cudaMemcpy(pix_out, pix_out_dev, sizeof(FILTER)*(x+2)*(y+2), cudaMemcpyDeviceToHost);
	cudaEventRecord(dev_stop, 0);
	cudaEventSynchronize(dev_stop);
	cudaEventElapsedTime(&d2hTime, dev_start, dev_stop);

    /* -----画像コピー----- */
    for(i=0;i<y;i++){
        for(j=0;j<x;j++){
			pix_dev[j + i*x] = (FILTER)pix_out[j+1 + (i+1)*(x+2)];
		}
	}

    /* compare cpu output and gpu output */
    for(i=0;i<y;i++){
        for(j=0;j<x;j++){
			if(pix_dev[j + i*x] != output[j + i*x]){
				printf("error\n");
				break;
			}
		}
	}

    /* output gpu results */
	fp = fopen("output.pgm", "w");
	fprintf(fp, "P2\n");
	fprintf(fp, "# CREATOR: GIMP PNM Filter Version 1.1\n");
	fprintf(fp, "%d %d\n", x, y);
	fprintf(fp, "255\n");
    for(i=0;i<y;i++){
        for(j=0;j<x;j++){
			if(pix_dev[j + i*x] > 255){
			    fprintf(fp, "255\n");
			}else if(pix_dev[j + i*x] < 0){
				fprintf(fp, "0\n");
			}else{
				fprintf(fp, "%d\n", pix_dev[j + i*x]);
			}
		}
	}
	fclose(fp);

	LONGLONG span = end.QuadPart - start.QuadPart;
	double sec = (double)span / (double)frequency.QuadPart;

	printf("cpu          : %lf [ms]\n", sec*1000);
	printf("HostToDevice : %f [ms]\n", h2dTime);
	printf("gpu          : %f [ms]\n", elapsedTime);
	printf("deviceToHost : %f [ms]\n", d2hTime);
	printf("gpuTotal     : %f [ms]\n", h2dTime + elapsedTime + d2hTime);

    free(pix);
    free(output);
}