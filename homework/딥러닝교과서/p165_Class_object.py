# 파이썬은 객체 지향 언어. 프로그래밍 세계에서 객체는 변수와 함수가 뭉쳐서 정리된 물건
# 리스트 내부에 변수와 함수를 가지고 있으며 상황에 따라 행동 방식을 변환
# 객체를 사용하면 프로그래머는 함수나 변수를 의식하고 관리할 필요 X
# 클래스 생성 시 세 가지 정의 : 생성자, 메서드, 멤버
# 생성자 : 클래스를 만들 때 자동으로 호출되는 특수 함수. 파이썬에서는 __init__
# 메서드 : 클래스가 갖는 처리. 함수
# 멤버 : 클래스가 가지는 값. 변수

# 1) 클래스(멤버와 생성자)
# 각각의 객체는 어떤 값을 가질지, 어떻게 처리할지 결정하기 위해 객체의 구조를 결정하는 설계도가 필요
# 이 설계도를 클래스라 함
# 클래스 호출 시, 작동하는 메서드를 생성자. __init__()로 정의, self를 생성자의 첫 번째 인수로 지정
class MyProduct:
    # 생성자를 수정
    def __init__(self, name, price, stock):
        # 인수를 멤버에 저장
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
product_1 = MyProduct("cake",500,20)
print(product_1.stock)      # 20
# cake라는 멤버의 가격은 500이고 stock는 20

#2) 메서드
# 메서드 정의 시 첫번째 인수로 self를 지정, 멤버 앞에 .self를 사용. 메서드를 호출 시 객체명.메서드명
class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
    def summary(self):
        message = "called summary()." + \
        "\n name: " + self.get_name() + \
        "\n price: " + str(self.price) + \
        "\n price: " + str(self.price) + \
        "\n sales: " + str(self.sales)
        print(message)
    # name을 반환하는 get_name()작성
    def get_name(self):
        return self.name
    # 인수만큼 price를 줄이는 discoun() 작성
    def discount(self,n):
        self.price -= n
product_2 = MyProduct("Phone", 30000, 100)
# 5000만큼 discount
product_2.discount(5000)
# product_2의 summary 출력
product_2.summary()
# called summary().
#  name: Phone
#  price: 25000
#  price: 25000
#  sales: 0

# 상속, 오버라이드, 슈퍼 클래스
# 다른 사람이 만든 클래스에 기능을 추가하고 싶을 때
# 부모 클래스의 메서드와 멤버를 덮어쓸 수 있음(오버라이드)
# 자식 클래스에서 부모 클래스의 메서드와 멤버 호출(슈퍼)
class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
    def summary(self):
        message = "called summary()." + \
        "\n name: " + self.get_name() + \
        "\n price: " + str(self.price) + \
        "\n price: " + str(self.price) + \
        "\n sales: " + str(self.sales)
        print(message)
    def get_name(self):
        return self.name
    def discount(self,n):
        self.price -= n

class MyProductSalesTax(MyProduct):
    def __init__(self, name, price, stock, tax_rate):
        # super()를 사용하여 부모 클래스 메서드 호출
        super().__init__(name, price, stock)
        self.tax_rate = tax_rate
    # 자식 클래스에서 부모 클래스의 get_name을 오버라이드
    def get_name(self):
        return self.name + "(소비세포함)"
    # 자식 클래스에 get_price_with_tax를 구현
    def get_price_with_tax(self):
        return int(self.price * (1+self.tax_rate))
    # 부모 클래스의 summary() 메서드 재정의, 소비세를 포함한 가격 출력
    def summary(self):
        message = "called summary().\n name: " + self.get_name() + \
        "\n price: " + str(self.get_price_with_tax()+0) + \
        "\n stock: " + str(self.stock) + \
        "\n sales: " + str(self.sales)
        print(message)
product_3 = MyProductSalesTax("Phone",30000,100,0.1)
print(product_3.get_name())
print(product_3.get_price_with_tax())
product_3.summary()
# called summary().
#  name: Phone(소비세포함)
#  price: 33000
#  stock: 100
#  sales: 0
