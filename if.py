#  -*-  coding = utf-8  -*-
#  @Time  :2021/8/20  8:35
#  @Author  : 小虎旻子
#  @File  :if.py.py
#  @Sofeware:  PyCharm
"""
    选择语句
"""
# sex = input("请输入性别:")
# if sex == "男":
#     print("您好，先生!")
# else:
#     print("您好,女士!")
#
# print("后续逻辑")


# 调试: 让程序中断，逐语句执行
#   -- 目的


# price = float(input("请输入商品单价:"))
# count = float(input("请输入商品数量:"))
# money = float(input("请输入金额:"))
# result = money - price * count
# if result >= 0:
#     print("应找回:"+str(result))
# else:
#     print("金额不足")


"""
    练习:在控制台中获取一个季度(春夏秋冬)
        显示相对应的月份
"""
#   如果前面条件满足，后续的条件就不在判断了
#   程序运行的效率就变高了，有互斥性
# season = input("请输入季度:")
# if season == "春":
#     print("1月,2月,3月")
# elif season == "夏":
#     print("4月,5月,6月")
# elif season == "秋":
#     print("7月,8月,9月")
# elif season == "冬":
#     print("10月,11月,12月")
# else:
#     print("输入月份有误，请重新输入")



"""
    练习2: 在控制台中录入一个数字，
        再录入一个运算符(+ - * /)
        最后再录入一个数字
        根据运算符计算两个数字
    要求:如果运算符不是加减乘除
         则提示"运算符错误"  
"""

# number1 = float(input(
#     "请输入第一个数字:"))
# operator = input("请输入运算符:")
# number2 = float(input(
#     "请输入第二个数字:"))
#
# if operator == "+":
#     print(number1 + number2)
# elif operator == "-":
#     print(number1 - number2)
# elif operator == "*":
#     print(number1 * number2)
# elif operator == "/":
#     print(number1 / number2)
# else:
#     print("运算符输入有误")



"""
    练习3: 在控制台中分别录入4个数字
           打印最大的数字
"""
# number1 = int(input("请输入第一个数字:"))
# number2 = int(input("请输入第二个数字:"))
# number3 = int(input("请输入第三个数字:"))
# number4 = int(input("请输入第四个数字:"))
# max_number = number1
# if number1 > number2:
#     max_number = number1
#     if number1 > number3:
#         max_number = number1
#         if number1 > number4:
#             max_number = number1
# elif number1 < number2:
#     max_number = number2
#     if number2 > number3:
#         max_number = number2
#         if number2 > number4:
#             max_number = number2
# elif number2 < number3:
#     max_number = number3
#     if number3 > number4:
#         max_number = number3
# elif number3 < number4:
#     max_number = number4
"""
    将第一个数字记在心里，然后与第二个比较
    如果第二个大于心中的数字，则心中记录第二个
    然后和第三个比较.......
"""
# number1 = int(input("请输入第一个数字:"))
# number2 = int(input("请输入第二个数字:"))
# number3 = int(input("请输入第三个数字:"))
# number4 = int(input("请输入第四个数字:"))
# # 假设第一个是最大值
# max_number = number1
# # 以此与后面进行比较
# if max_number < number2:
#     max_number = number2
# if max_number < number3:
#     max_number = number3
# if max_number < number4:
#     max_number = number4
# print(max_number)



"""
    在控制台中录入一个成绩，
    判断等级(优秀/良好/及格/不及格/输入有误)
"""
# score = int(input("请输入成绩"))
# if score >= 90 and score <= 100:
#     print("优秀")
# elif score >= 80 and score < 90:
#     print("良好")
# elif score >= 60 and score < 80:
#     print("及格")
# elif score >= 0 and score < 60:
#     print("不及格")
# else:
#     print("输入有误")

# if 90 <= score <= 100:
#     print("优秀")
# elif 80 <= score < 90:
#     print("良好")
# elif 60 <= score < 80:
#     print("及格")
# elif 0 <= score < 60:
#     print("不及格")
# else:
#     print("输入有误")



"""
    在控制台中获取一个月份
    打印天数，或者提示"提示有误"
    1 3 5 7 8 10 12 ---> 31天
    4 6 9 11 ---> 30天
    2 ---> 28天
"""
# month = int(input("请输入月份:"))
# if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
#     print("该月有31天")
# elif month == 4 or month == 6 or month == 9 or month == 11:
#     print("该月有30天")
# elif month == 2:
#     print("该月有28天")

# month = input("请输入月份:")
# if month < "1" or month >"12":
#     print("输入有误")
# elif month == "2":
#     print("28天")
# elif month == "4" or month == "6" or month == "9" or month == "11":
#     print("30天")
# else:
#     print("31天")


"""
    练习:在控制台中获取一个整数
        如果是偶数为变量state赋值"偶数"
        否则赋值为”奇数“
"""
# number = int(input("获取一个整数:"))
# if number % 2 == 0:
#     state = "偶数"
# else:
#     state = "奇数"
# print(state)

# state = "偶数" if number % 2 == 0 else "奇数"
# print(state)

"""
    练习:在控制台中录入一个年份
        如果是闰年，给变量day赋值29，否则赋值28
"""
# if (year % 4 == 0) and (year % 100 != 0) or (year % 400 == 0):
#     day = 29
# else:
#     day = 28
# print(day)
year = int(input("获取一个年份:"))
day = 29 if (year % 4 == 0) and (year % 100 != 0) or (year % 400 == 0) else 28
print(day)



