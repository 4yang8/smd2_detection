class ScoreManager:
    def __init__(self):
        self.user_scores = {}  # 사용자 점수는 빈 딕셔너리로 시작

    def get_user_score(self, user_id):
        # 사용자 점수를 가져올 때, 점수가 없으면 자동으로 -2로 초기화된 점수 반환
        if user_id not in self.user_scores:
            self.user_scores[user_id] = {i: -2 for i in range(1, 10)}  # 1~9 항목에 대해 -1로 초기화
        return self.user_scores[user_id]

    def update_score(self, user_id, score):
        # 점수 업데이트: 기존 점수를 완전히 덮어씀
        if user_id not in self.user_scores:
            self.user_scores[user_id] = {i: -2 for i in range(1, 10)}  # 초기화가 필요하면 자동으로 초기화

        # 기존 점수는 완전히 새 점수로 덮어씀
        for key, value in score.items():
            self.user_scores[user_id][key] = value
