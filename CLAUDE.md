# 딥러닝 추천 알고리즘 분야에 vectorDB 전용 processor 도입을 위한 benchmark 개발

# 1. 목표, 요구사항
우리 회사는 vectorDB 전용 processor (VDPU;https://www.hellot.net/news/article.html?no=99889) 를 만들었습니다. 이를 쿠팡과 같은 쇼핑몰 회사에 팔기위해 VDPU를 적용하게 되면 vector연산의 속도 향상으로 인해 큰 비용절감 효과를 얻을수 있다고 홍보를 하고 싶습니다. 이에대한 근거 데이터를 얻기위한 benchmark 개발이 필요한 상황입니다. 추천에서 널리 사용되는 two-tower 모델을 기반으로 vector 연산(ANN) 비용을 직/간접적으로 반영하여 수치화할수있는 benchmark를 개발해야합니다.
여러 오픈소스 커머셜 추천 benchmark를 조사하여 우리의 benchmark 메트릭 설계/개발/문서화를 진행해 주세요.