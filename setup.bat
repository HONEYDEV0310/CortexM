@echo off
echo 가상환경(.venv)을 생성하는 중...
python -m venv .venv

echo 가상환경을 활성화하고 라이브러리를 설치하는 중...
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

echo 모든 설치가 완료되었습니다!
pause