'''
--url--
https://www.acmicpc.net/problem/11654

--title--
11654번: 아스키 코드

--problem_description--
알파벳 소문자, 대문자, 숫자 0-9중 하나가 주어졌을 때, 주어진 글자의 아스키 코드값을 출력하는 프로그램을 작성하시오.

--problem_input--
알파벳 소문자, 대문자, 숫자 0-9 중 하나가 첫째 줄에 주어진다.

--problem_output--
입력으로 주어진 글자의 아스키 코드 값을 출력한다.

'''
#1. 숫자, 글자 다 받게끔 input
#2. '아스키코드 파이썬' 구글링
#3. 문자와 숫자가 반환해주는 명령어가 다르기에 if else문 사용

text = input()

if type(text)==int:
    print(chr(text))
else:
    print(ord(text))
