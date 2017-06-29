""" testing different kinds of sorts """
import timeit
from random import *
from copy import copy
import numpy as np
import matplotlib.pyplot as plt

""" implements bubble sort """
def bubbleSort(A):
    for i in range(len(A)):
        for k in range(len(A) - 1, i, -1):
            if A[k] < A[k - 1]:
                swap(A, k, k - 1)
    return A

def selectionSort(array):
    for i in range(0, len(array) - 1):
        smallSub = i
        for j in range(i + 1, len(array) - 1):
            if array[j] < array[smallSub]:
                smallSub = j
        temp = array[i]
        array[i] = array[smallSub]
        array[smallSub] = temp
    return array

def insertionSort(aList):
    for i in range(1, len(aList)):
        tmp = aList[i]
        k = i
        while k > 0 and tmp < aList[k - 1]:
            aList[k] = aList[k - 1]
            k -= 1
        aList[k] = tmp

def mergeSort(array):
    if len(array) == 1:
        return array

    list1 = []
    list2 = []

    for i in range(0, len(array)):
        if i < len(array) / 2:
            list1.append(array[i])
        else:
            list2.append(array[i])

    list1 = mergeSort(list1)
    list2 = mergeSort(list2)

    return merge(list1, list2)

def qsort(a):
    if len(a) <= 1:
        return a
    else:
        q = choice(a)
        return qsort([elem for elem in a if elem < q]) + [q] * a.count(q) + qsort([elem for elem in a if elem > q])

def merge(array1, array2):
    array3 = []
    count1 = 0
    count2 = 0

    while count1 != len(array1) and count2 != len(array2):
        if array1[count1] > array2[count2]:
            array3.append(array2[count2])
            count2 += 1
        else:
            array3.append(array1[count1])
            count1 += 1
    while count1 != len(array1):
        array3.append(array1[count1])
        count1 += 1
    while count2 != len(array2):
        array3.append(array2[count2])
        count2 += 1
    return array3


def heapSort(aList):
    # convert aList to heap
    length = len(aList) - 1
    leastParent = length / 2
    for i in range(int(leastParent), -1, -1):
        moveDown(aList, i, length)

    # flatten heap into sorted array
    for i in range(length, 0, -1):
        if aList[0] > aList[i]:
            swap(aList, 0, i)
            moveDown(aList, 0, i - 1)

    return aList

def moveDown(aList, first, last):
    largest = 2 * first + 1
    while largest <= last:
        # right child exists and is larger than left child
        if (largest < last) and (aList[largest] < aList[largest + 1]):
            largest += 1

        # right child is larger than parent
        if aList[largest] > aList[first]:
            swap(aList, largest, first)
            # move down to largest child
            first = largest
            largest = 2 * first + 1
        else:
            return # force exit

def swap(A, x, y):
    tmp = A[x]
    A[x] = A[y]
    A[y] = tmp

def comparisonCountingSort(array):
    count = []
    sort = []

    for i in range(0, len(array) - 1):
        count.append(0)
        sort.append(0)

    for i in range(0, len(array) - 2):
        for j in range(i + 1, len(array) - 1):
            if array[i] < array[j]:
                count[j] += 1
            else:
                count[i] += 1

    for i in range(0, len(array) - 1):
        sortedArray[count[i]] = array[i]

    return sortedArray

randomArray = []
sortedArray = []
backwardsArray = []
stepsArray = []
upper = 10000

for ind in range(0, upper):
    randomArray.append(randint(0, 100))
    sortedArray.append(ind)
    backwardsArray.append(upper - ind)
    stepsArray.append(randint(0, 10) * 1000)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def quickSortPrep(array):
    return qsort(array)

def listAppend(bubble, insertion, selection, merge, quick, heap, count, array, length):
    catArray = array[:length]

    bubbleWrap = wrapper(bubbleSort, copy(catArray))
    bubble.append(timeit.timeit(bubbleWrap, number=1))

    selectionWrap = wrapper(selectionSort, copy(catArray))
    selection.append(timeit.timeit(selectionWrap, number=1))

    insertionWrap = wrapper(insertionSort, copy(catArray))
    insertion.append(timeit.timeit(insertionWrap, number=1))

    mergeWrap = wrapper(mergeSort, copy(catArray))
    merge.append(timeit.timeit(mergeWrap, number=1))

    quickWrap = wrapper(quickSortPrep, copy(catArray))
    quick.append(timeit.timeit(quickWrap, number=1))

    heapWrap = wrapper(heapSort, copy(catArray))
    heap.append(timeit.timeit(heapWrap, number=1))

    countWrap = wrapper(comparisonCountingSort, copy(catArray))
    count.append(timeit.timeit(countWrap, number=1))

def resetLists(bubble, insertion, selection, merge, quick, heap, count):
    bubble[:] = []
    insertion[:] = []
    selection[:] = []
    merge[:] = []
    quick[:] = []
    heap[:] = []
    count[:] = []

def testSorts():
    bubbleTime = []
    insertionTime = []
    selectionTime = []
    mergeTime = []
    quickTime = []
    heapTime = []
    countTime = []
    times = []

    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, randomArray, 10)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, sortedArray, 10)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, backwardsArray, 10)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, stepsArray, 10)

    times.append(
        [copy(bubbleTime),
        copy(insertionTime),
        copy(selectionTime),
        copy(mergeTime),
        copy(quickTime),
        copy(heapTime),
        copy(countTime)]
    )
    resetLists(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime)

    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, sortedArray, 20)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, randomArray, 20)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, backwardsArray, 20)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, stepsArray, 20)

    times.append(
        [copy(bubbleTime),
        copy(insertionTime),
        copy(selectionTime),
        copy(mergeTime),
        copy(quickTime),
        copy(heapTime),
        copy(countTime)]
    )
    resetLists(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime)

    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, sortedArray, 100)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, randomArray, 100)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, backwardsArray, 100)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, stepsArray, 100)

    times.append(
        [copy(bubbleTime),
        copy(insertionTime),
        copy(selectionTime),
        copy(mergeTime),
        copy(quickTime),
        copy(heapTime),
        copy(countTime)]
    )
    resetLists(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime)

    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, sortedArray, 1000)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, randomArray, 1000)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, backwardsArray, 1000)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, stepsArray, 1000)

    times.append(
        [copy(bubbleTime),
        copy(insertionTime),
        copy(selectionTime),
        copy(mergeTime),
        copy(quickTime),
        copy(heapTime),
        copy(countTime)]
    )
    resetLists(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime)

    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, sortedArray, 10000)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, randomArray, 10000)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, backwardsArray, 10000)
    listAppend(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime, stepsArray, 10000)

    times.append(
        [copy(bubbleTime),
        copy(insertionTime),
        copy(selectionTime),
        copy(mergeTime),
        copy(quickTime),
        copy(heapTime),
        copy(countTime)]
    )
    resetLists(bubbleTime, insertionTime, selectionTime, mergeTime, quickTime, heapTime, countTime)

    #populateChart(times[0])
    #populateChart(times[1])
    #populateChart(times[2])
    #populateChart(times[3])
    #populateChart(times[4])

    #plotlogn(times[3])

def plotlogn(timeList):
    groups = 4
    fig, ax = plt.subplots()
    index = np.arange(groups)
    bar_width = .25

    mergeRect = plt.bar(
        index + bar_width * 3,
        timeList[3],
        bar_width,
        color='orange',
        label='Merge'
    )

    quickRect = plt.bar(
        index + bar_width * 4,
        timeList[4],
        bar_width,
        color='red',
        label='Quick'
    )

    heapRect = plt.bar(
        index + bar_width * 5,
        timeList[5],
        bar_width,
        color='purple',
        label='Heap'
    )

    plt.xlabel('Sorting Algorithm')
    plt.ylabel('Sorting Time')
    plt.title('Comparison of Log(n) Sorting Algorithms')
    plt.xticks(
        index + bar_width * 3,
        ('Random', 'Sorted', 'Backwards', 'Steps'))
    plt.legend()

    plt.tight_layout()
    plt.show()

def populateChart(timeList):
    groups = 4
    fig, ax = plt.subplots()
    index = np.arange(groups)
    bar_width = .125

    bubbleRect = plt.bar(
        index,
        timeList[0],
        bar_width,
        color='g',
        label='Bubble'
    )

    selectionRect = plt.bar(
        index + bar_width,
        timeList[1],
        bar_width,
        color='b',
        label='Selection'
    )

    insertionRect = plt.bar(
        index + bar_width * 2,
        timeList[2],
        bar_width,
        color='y',
        label='Insertion'
    )

    mergeRect = plt.bar(
        index + bar_width * 3,
        timeList[3],
        bar_width,
        color='orange',
        label='Merge'
    )

    quickRect = plt.bar(
        index + bar_width * 4,
        timeList[4],
        bar_width,
        color='red',
        label='Quick'
    )

    heapRect = plt.bar(
        index + bar_width * 5,
        timeList[5],
        bar_width,
        color='purple',
        label='Heap'
    )

    countRect = plt.bar(
        index + bar_width * 6,
        timeList[6],
        bar_width,
        color='black',
        label='Count'
    )

    plt.xlabel('Sorting Algorithm')
    plt.ylabel('Sorting Time')
    plt.title('Comparison of Sorting Algorithms')
    plt.xticks(
        index + bar_width * 3,
        ('Random', 'Sorted', 'Backwards', 'Steps'))
    plt.legend()

    plt.tight_layout()
    plt.show()

testSorts()