package osu.crowd_ml.utils;

/*
Copyright 2017 Crowd-ML team


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
*/

public final class ArrayUtils {
    private ArrayUtils(){}

    // Merge Sort
    public static void sort(int[] array) {
        quickSort(array, 0, array.length - 1);
    }

    private static void quickSort(int[] a, int low, int high){
        int index = partition(a, low, high);
        if (low < index - 1){
            quickSort(a, low, index - 1);
        }
        if (index < high){
            quickSort(a, index, high);
        }
    }

    private static int partition(int[] a, int low, int high) {
        int pivot = a[(low + high) / 2];
        while (low <= high) {
            while(a[low] < pivot) low++;
            while(a[high] > pivot) high--;

            if (low <= high) {
                swap(a, low, high);
                low++;
                high--;
            }
        }
        return low;
    }

    private static void swap(int[] a, int low, int high) {
        int temp = a[low];
        a[low] = a[high];
        a[high] = temp;
    }

    public static int binarySearch(int[] array, int query) {
        return binarySearch(array, query, 0, array.length - 1);
    }

    private static int binarySearch(int[] a, int query, int low, int high) {
        if (low > high){
            return -1;
        }

        int index = (low + high) / 2;

        if (a[index] > query){
            index = binarySearch(a, query, 0, index - 1);
        } else if (a[index] < query) {
            index = binarySearch(a, query, index + 1, high);
        }
        return index;
    }
}
