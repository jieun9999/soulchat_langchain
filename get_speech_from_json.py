import json
import os

def extract_texts_from_json_file(file_path):
  """
  주어진 경로의 단일 JSON 파일을 읽어 "utterances" 리스트 내의 모든 "text" 값을 추출하여 리스트로 반환합니다.

  Args:
    file_path (str): 읽어올 JSON 파일의 경로.

  Returns:
    list: 추출된 "text" 문자열들을 담은 리스트. 오류 발생 시 빈 리스트를 반환합니다.
  """
  texts = []
  try:
    # 파일을 utf-8 인코딩으로 엽니다.
    with open(file_path, 'r', encoding='utf-8') as f:
      # JSON 파일을 파이썬 객체로 로드합니다.
      data = json.load(f)

    # 'utterances' 키가 존재하고, 그 값이 리스트인지 확인합니다.
    if 'utterances' in data and isinstance(data['utterances'], list):
      # 'utterances' 리스트 내의 각 항목(딕셔너리)에 대해 반복합니다.
      for utterance in data['utterances']:
        # 각 딕셔너리 안에 'text' 키가 있고, 그 값이 비어있지 않은지 확인합니다.
        if isinstance(utterance, dict) and 'text' in utterance and utterance['text']:
          # 'text' 키의 값을 texts 리스트에 추가합니다.
          texts.append(utterance['text'])
    # else:
        # 주석 처리: 개별 파일 처리 시 경고 메시지 비활성화 (필요시 활성화)
        # print(f"경고: '{os.path.basename(file_path)}' 파일에 'utterances' 키가 없거나 리스트 형식이 아닙니다.")
        pass

  # FileNotFoundError는 상위 함수에서 디렉토리 순회 중 처리되므로 여기서 무시해도 될 수 있습니다.
  # 다만, 개별 파일 접근 권한 문제 등 다른 I/O 오류 가능성은 있습니다.
  except json.JSONDecodeError:
    print(f"오류: JSON 파일을 파싱하는 중 오류가 발생했습니다 - {os.path.basename(file_path)}")
  except Exception as e:
    # 파일 권한 문제 등 다른 예외 처리
    print(f"파일 처리 중 예상치 못한 오류 발생 ({os.path.basename(file_path)}): {e}")

  return texts

def extract_texts_from_directory(directory_path, limit=500):
  """
  주어진 디렉토리 내의 모든 JSON 파일을 순회하며 "text" 값을 추출하여,
  목표 개수(limit)에 도달할 때까지 리스트에 추가하고 반환합니다.

  Args:
    directory_path (str): JSON 파일들이 있는 디렉토리 경로.
    limit (int): 추출할 최대 문장 개수. 기본값은 500.

  Returns:
    list: 추출된 "text" 문자열들을 담은 리스트. 최대 limit 개수만큼 포함됩니다.
  """
  all_texts = []
  processed_files = 0

  try:
    # 디렉토리 내의 모든 파일/폴더 목록을 가져옵니다.
    # 순서를 보장하고 싶다면 listdir 결과를 정렬할 수 있습니다: sorted(os.listdir(directory_path))
    items = os.listdir(directory_path)
  except FileNotFoundError:
    print(f"오류: 디렉토리를 찾을 수 없습니다 - {directory_path}")
    return [] # 디렉토리가 없으면 빈 리스트 반환
  except Exception as e:
    print(f"오류: 디렉토리 접근 중 오류 발생 - {e}")
    return []

  print(f"'{directory_path}' 디렉토리에서 최대 {limit}개의 문장을 추출합니다...")

  # 디렉토리 내의 각 항목에 대해 반복합니다.
  for item_name in items:
    # 항목의 전체 경로를 만듭니다.
    full_path = os.path.join(directory_path, item_name)

    # 해당 경로가 파일이고, 확장자가 .json (대소문자 무시)인지 확인합니다.
    if os.path.isfile(full_path) and item_name.lower().endswith('.json'):
      # 현재 파일에서 텍스트를 추출합니다.
      texts_from_file = extract_texts_from_json_file(full_path)
      processed_files += 1

      # 추출된 텍스트가 있다면 all_texts 리스트에 추가합니다.
      if texts_from_file:
        all_texts.extend(texts_from_file)

        # 목표 개수에 도달했거나 초과했는지 확인합니다.
        if len(all_texts) >= limit:
          print(f"\n목표 문장 수({limit})에 도달했습니다. {processed_files}개 파일 처리 후 중단합니다.")
          # 목표 개수만큼만 잘라서 반환합니다.
          return all_texts[:limit]

  # 모든 파일을 처리했지만 목표 개수에 도달하지 못한 경우
  print(f"\n디렉토리 내 모든 JSON 파일({processed_files}개) 처리를 완료했습니다.")
  return all_texts # 현재까지 모인 모든 텍스트 반환 (limit 이하일 것임)

# --- 실행 부분 ---

# JSON 파일들이 있는 디렉토리 경로 설정
# target_directory = "/workspace/hdd/1.korean_SNS_multiTurn_conversation_data/3.openData/1.data/Training/2.labellingData/DailyTrend1.health_and_foodAndDrink"
# 추출할 문장 개수 목표 설정
# sentence_limit = 500

# # 함수를 호출하여 디렉토리에서 "text" 값들을 추출합니다.
# final_extracted_texts = extract_texts_from_directory(target_directory, sentence_limit)

