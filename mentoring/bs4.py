from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus

base_url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
plus_url = input('검색어를 입력하세요 : ')

# quote_plus==URL로 이동하기 위한 쿼리 문자열을 만들 때,
# HTML 폼값을 인용하는 데 필요한 대로 스페이스를 + 부호로 치환
url = base_url + quote_plus(plus_url)

html = urlopen(url).read()
soup = bs(html, "html.parser")
# _img라는 클래스를 가져와라
img = soup.select("_img")

print(img[0])

n = 1
for i in img:
    img_url = i['data-source']
    with urlopen(img_url) as url:
        with open(plus_url+str(n) + '.jpg', 'wb') as load:
            img = url.read()
            load.write(img)
    n += 1

print('다운로드완료')