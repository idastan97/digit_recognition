
from random import choice as ch

numbers = []


def new_num(m, n):
    c_num = m.copy()
    for i in range(len(n)):
        d = ch(list(range(4)))
        while c_num[d] != '':
            d = ch(list(range(4)))
        c_num[d] = ch(n)
        n.remove(c_num[d])
    count = c_num.count('')
    if count > 0:
        for i in range(4):
            if c_num[i] == '':
                c_num[i] = str(ch(list(range(10))))
    return c_num


def check(g_arr, r_arr):
    bk = 0
    kr = 0
    t_b = []
    t_k = []
    for i in range(4):
        if g_arr[i] == r_arr[i]:
            bk += 1
            t_b.append(g_arr[i])
        else:
            t_b.append('')
        if g_arr[i] != r_arr[i]:
            if g_arr[i] in r_arr:
                kr += 1
                t_k.append(g_arr[i])
    return bk, kr, new_num(t_b, t_k)


print('\nСекретный уровень!')
g_num = list(input('\n:'))
r_num = str(ch(list(range(1000, 10000))))
while g_num != r_num:
    if len(g_num) == 4:
        bk, kr, r_num = check(g_num, r_num)
        print('Быки:', bk, '\nКоровы:', kr)
    elif len(g_num) > 4:
        print('цифра слишком большая')
    elif len(g_num) < 4:
        print('цифра слишком маленькая')
    g_num = list(input('\n:'))

print(
    '''
Мой поздравления!
Ты прошел секретный уровень!
    '''
)