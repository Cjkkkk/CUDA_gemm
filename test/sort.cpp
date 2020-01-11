void bitonic_sort(bool up, float* x, int n);
void bitonic_merge(bool up, float* x, int n);
void bitonic_compare(bool up, float* x, int n);

void bitonic_sort(bool up, float* x, int n) {
    if (n <= 1)
        return;
    else {
        bitonic_sort(true, x, n / 2);
        bitonic_sort(false, x + n / 2, n / 2);
        bitonic_merge(up, x, n);
    }
}

void bitonic_merge(bool up, float* x, int n) {
    if (n == 1)
        return;
    else {
        bitonic_compare(up, x, n);
        bitonic_merge(up, x, n / 2);
        bitonic_merge(up, x + n / 2, n / 2);
    }
}

void bitonic_compare(bool up, float* x, int n) {
    int dist = n / 2;
    for (int i = 0 ; i < dist ; i ++) {
        if ((x[i] > x[i + dist]) == up) {
            float temp = x[i];
            x[i] = x[i + dist];
            x[i + dist] = temp;
        }
    }
}

int main() {
    float* arr = (float*) malloc(sizeof(float) * 8);
    arr[0] = 8;
    arr[1] = 7;
    arr[2] = 10;
    arr[3] = 6;
    arr[4] = 9;
    arr[5] = 3;
    arr[6] = 4;
    arr[7] = 12;
    bitonic_sort(true, arr, 8);
    for ( int i = 0 ; i < 8 ; i ++ ) {
        printf("%f ", arr[i]);
    }

}