#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

int continue_choose();
int load_image(double image[16][784], char name_output[16][128]);
int load_weights(double w1[128][784], double w2[10][128], double b1[128], double b2[10]);

double ReLU(double);
void SoftmaxFunc(double *, double *, int);

int MLP(double image[784],
        double w1[128][784],
        double w2[10][128],
        double b1[128],
        double b2[10],
        char img_name[128]);
int layer1(double weights[128][784], double bias[128], double source[784], double dest[128]);
int layer2(double weights[10][128], double bias[10], double source[128], double dest[10]);
int collect_layer(double *, double *);

int main()
{

    double image[16][784];
    double w1[128][784];
    double w2[10][128];
    double b1[128];
    double b2[10];
    char img_name[16][128];
    int vaild_file_count;
    int weight_loading_status;

    weight_loading_status = load_weights(w1, w2, b1, b2);
    if (weight_loading_status)
    {
        return -1;
    }

    while (1)
    {
        vaild_file_count = load_image(image, img_name);
        for (int i = 0; i < vaild_file_count; i++)
        {
            MLP(image[i], w1, w2, b1, b2, img_name[i]);
        }

        if (vaild_file_count == -1)
        {
            return 0;
        }
        else if(!continue_choose())
        {
            return 0;
        }
    }
}

int load_weights(double w1[128][784], double w2[10][128], double b1[128], double b2[10])
{

    FILE *w1_f = fopen("./Input files/W1.txt", "r"); // open file
    FILE *w2_f = fopen("./Input files/W2.txt", "r"); // open file
    FILE *b1_f = fopen("./Input files/B1.txt", "r"); // open file
    FILE *b2_f = fopen("./Input files/B2.txt", "r"); // open file
    if (w1_f == NULL || w2_f == NULL || b1_f == NULL || b2_f == NULL)
    {
        printf("Cannot load weight file,\n Please make sure you place weight file in ./Input files/ folder.\n And named them in right form\n");
        return -1;
    }

    for (int i = 0; i < 784; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            ((double)fscanf((FILE *)w1_f, "%lf", &w1[j][i]));
        }
    }

    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            ((double)fscanf((FILE *)w2_f, "%lf", &w2[j][i]));
        }
    }

    for (int i = 0; i < 128; i++)
    {
        ((double)fscanf((FILE *)b1_f, "%lf", &b1[i]));
    }
    for (int i = 0; i < 10; i++)
    {
        ((double)fscanf((FILE *)b2_f, "%lf", &b2[i]));
    }

    fclose(w1_f);
    fclose(w2_f);
    fclose(b1_f);
    fclose(b2_f);
    return 0;
}

int continue_choose()
{
    char operate;
    printf("Do you want to continue? please input [Y or N]:");
    operate = getchar();
    fflush(stdin);
    if (operate == 'Y' || operate == 'y')
    {
        return 1;
    }
    else if (operate == 'N' || operate == 'n')
    {
        printf("Bye");
        return 0;
    }
    else
    {
        printf("Invild input.\n");
        return continue_choose();
    }
}

int load_image(double image[16][784], char name_output[16][128])
{
    char file_name[128];
    char query[16][128];
    int operate = 0;
    int file_count = 0;
    FILE *img_f;

    printf("Please input filename(ONLY predict 16 files in a query):");
    fgets(file_name, 127, stdin);
    if (file_name[strlen(file_name) - 1] == '\n')
        file_name[strlen(file_name) - 1] = '\0';
    fflush(stdin);

    char char_temp = 'a';
    int file_name_cursor = 0;
    for (int i = 0; i < 128; i++)
    {
        char_temp = file_name[i];

        if (char_temp == ',')
        {
            query[file_count][file_name_cursor] = '\0';
            file_name_cursor = 0;
            file_count++; // count files
            continue;
        }

        query[file_count][file_name_cursor] = file_name[i];
        file_name_cursor++;

        if (file_count == 16 || char_temp == '\0')
        {
            break;
        }
    }
    file_count++;

    for (int i = 0; i < 16; i++)
    {
        strcpy(name_output[i], query[i]);
    }

    int vaild_file = 0;
    for (int n = 0; n < file_count; n++)
    {
        char name_current[128];
        strcpy(name_current, query[n]);

        img_f = fopen(name_current, "rb"); // open file

        if (img_f == NULL)
        {
            printf("Invaild file(%s).\n", name_current);
            continue;
        }

        for(int i = 0; i < 43; i++) fgetc((FILE *)img_f); // initial the file pointer

        for (int i = 0; i < 784; i++) // read the picture
        {
            image[vaild_file][i] = ((double)fgetc((FILE *)img_f)) / 255;

            /*
            //This is images printing code
            printf("Image:\n");
            if (image[i] <= 0.001)
            {
                printf("0");
            }
            else
            {
                printf(".");
            }
            if (i%28 == 0)
            {
                printf("\n");
            }*/
        }

        fclose(img_f); // close the file
        vaild_file++;
    }
    if (vaild_file == 0)
    {
        if (continue_choose())
        {
            return load_image(image, name_output);
        }
        else
            return -1;
    }

    return vaild_file;
}

int MLP(double image[784],
        double w1[128][784],
        double w2[10][128],
        double b1[128],
        double b2[10],
        char img_name[128])
{
    double layer1_output[128];
    double layer2_output[10];
    double prediction_output[10];

    layer1(w1, b1, image, layer1_output);
    layer2(w2, b2, layer1_output, layer2_output);
    collect_layer(layer2_output, prediction_output);

    int result = 0;
    for (int i = 0; i < 10; i++)
    {
        if (prediction_output[i] > prediction_output[result])
            result = i;
    }

    printf("(%s)prediction output:%d\n", img_name, result);
}

double ReLU(double input)
{
    if (input > 0)
        return input;

    else
        return 0;
}

void SoftmaxFunc(double *src, double *dst, int length)
{
    int i = 0;
    double sum = 0.0;
    for (i = 0; i < length; i++)
    {
        dst[i] = exp(src[i]);
    }
    for (i = 0; i < length; i++)
    {
        sum = sum + dst[i];
    }
    for (i = 0; i < length; i++)
    {
        dst[i] = dst[i] / sum;
    }
}

int layer1(double weights[128][784], double bias[128], double source[784], double dest[128])
{
    int length = 784;
    int node_count = 128;
    for (int i = 0; i < node_count; i++)
    {
        double temp = 0;
        for (int j = 0; j < length; j++)
        {
            double w = weights[i][j];
            double s = source[j];
            temp = temp + weights[i][j] * source[j];
        }
        dest[i] = ReLU(temp + bias[i]);
    }
    return *dest;
}

int layer2(double weights[10][128], double bias[10], double source[128], double dest[10])
{
    int length = 128;
    int node_count = 10;
    for (int i = 0; i < node_count; i++)
    {
        double temp = 0;
        for (int j = 0; j < length; j++)
        {
            temp = temp + weights[i][j] * source[j];
        }
        dest[i] = ReLU(temp + bias[i]);
    }
    return *dest;
}

int collect_layer(double source[10], double result[10])
{
    SoftmaxFunc(source, result, 10);
    return *result;
}