#include <iostream>
#include <stdio.h>

// Embedding 权重顺序为 (Token, CH)
void read_bin(const char* bin_file, int numb){
    uint16_t* embeddings = (uint16_t*)malloc(sizeof(uint16_t)*4096*65024);
    FILE *fp1 = fopen(bin_file, "rb");
    if (fp1==NULL) {
        printf("Can't open file: %s\n",bin_file);
    }
    fread(embeddings, sizeof(uint16_t), 4096*65024, fp1);
    fclose(fp1);
    int block = 65024 / numb;
    char file_name[100];
    for (int i = 0; i < numb; i++) {
        sprintf(file_name, "data/Embedding_%02d-of-%02d.bin", i+1, numb);
        FILE *fp = fopen(file_name, "wb");
        if (fp==NULL) {
            printf("Can't open file: %s\n",file_name);
        }
        fwrite(&embeddings[block*4096*i], sizeof(uint16_t), block*4096, fp);
        fclose(fp);
    }
}

int main(void) {
    read_bin("Embedding_Weight.bin", 16);
}
