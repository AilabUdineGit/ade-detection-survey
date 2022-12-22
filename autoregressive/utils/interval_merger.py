def merge_overlaps(spans):
    spans.sort(key = lambda x: x[0])
    new_spans= []
    i = 0
    while i < len(spans)-1:
        s1,e1 = spans[i] 
        s2,e2 = spans[i+1]
        if len(set(range(s1,e1)).intersection(set(range(s2,e2)))) != 0:
            spans[i+1] = (s1, max(e1,e2))
        else:
            new_spans.append((s1,e1))
        i += 1

    new_spans.append(spans[-1])
    return new_spans


if __name__=="__main__":
    a = [(36, 50), (52, 61), (63, 80), (86, 94), (124, 130), (130, 136), (140, 146), (146, 155), (164, 173), (197, 205), (209, 215),
(237, 257), (265, 275)]
    print(a)
    a = merge_overlaps(a)
    print(a)

    


class IntervalMerger(object):
    def merge(self, intervals):
        try:
            intervals = [[a,b,c] for (a,b,c) in intervals]
        except:
            intervals = [[a,b] for (a,b) in intervals]
        if len(intervals) == 0:
            return []
        self.quicksort(intervals,0,len(intervals)-1)
        stack = []
        stack.append(intervals[0])
        for i in range(1,len(intervals)):
            last_element= stack[len(stack)-1]
            if last_element[1] >= intervals[i][0]:
                last_element[1] = max(intervals[i][1],last_element[1])
                stack.pop(len(stack)-1)
                stack.append(last_element)
            else:
                stack.append(intervals[i])
        return [tuple(s) for s in stack]

    def partition(self,array,start,end):
        pivot_index = start
        for i in range(start,end):
            if array[i][0]<=array[end][0]:
                array[i],array[pivot_index] =array[pivot_index],array[i]
                pivot_index+=1
        array[end],array[pivot_index] =array[pivot_index],array[end]
        return pivot_index

    def quicksort(self,array,start,end):
        if start<end:
            partition_index = self.partition(array,start,end)
            self.quicksort(array,start,partition_index-1)
            self.quicksort(array, partition_index + 1, end)

