import sys
import numpy as np
import optparse


def parse_arg():
    parser = optparse.OptionParser()
    parser.add_option('-a', dest='action')
    parser.add_option('-f', dest='name')
    parser.add_option('-c', dest='pred')
    parser.add_option('-n', dest='num')
    (options, args) = parser.parse_args()
    return options


def gen_nums(N):
    return np.random.randint(-100, 100, size=N)


def qsort(nums):
    less, equal, greater = [], [], []
    if len(nums) > 1:
        pivot = nums[0]
        for x in nums:
            if x < pivot:
                less.append(x)
            if x == pivot:
                equal.append(x)
            if x > pivot:
                greater.append(x)
        return qsort(less) + equal + qsort(greater)
    else:
        return nums


def main(name, num, act, pred):
    if act == 'gen':
        nums = gen_nums(num)
        correct = qsort(nums)
        np.array(nums).astype(np.int32).tofile(open('{}.in'.format(name), 'wb'))
        np.array(correct).astype(np.int32).tofile(open('{}.out'.format(name), 'wb'))
    elif act == 'check':
        nums = np.fromfile(open('{}.out'.format(pred), 'rb'), dtype=np.int32)
        correct = np.fromfile(open('{}'.format(name), 'rb'), dtype=np.int32)
        ok = True
        for i in range(num):
            if nums[i] != correct[i]:
                ok = False
                break
        if ok:
            print('correct!')
        else:
            print('not correct!')
            print('correct: {}'.format(correct))
            print('but: {}'.format(nums))

if __name__ == '__main__':
    opt = parse_arg()
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['name'] = opt.name
        kwargs['num'] = int(opt.num)
        kwargs['act'] = opt.action
        kwargs['pred'] = opt.pred
    main(**kwargs)
