import konlpy
from konlpy.tag import Okt


okt = Okt()
print(okt.morphs("여기는 제주대학교 경영정보학과 입니다"))

result = okt.morphs("여기는 제주대학교 경영정보학과 입니다")
result2 = okt.morphs("오늘의 날씨는 좋습니다")
