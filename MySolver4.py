import sys
import time
import re
from copy import deepcopy

# 增加遞歸深度
sys.setrecursionlimit(5000)

# 常量設定
UNKNOWN = -1
EMPTY = 0
FILLED = 1

# 給每題的預算時間 (秒)
MAX_TIME_PER_PUZZLE = 15.0

# ======================================================
# 輔助函數
# ======================================================
def is_valid_hint(hints, N=25):
    """檢查一組 hint 是否在 N 格內合法"""
    if not hints: return True 
    min_len = sum(hints) + len(hints) - 1
    return min_len <= N

# ======================================================
# 智慧解析器 (Smart Parser v11.0 - 安全切割版)
# ======================================================
def parse_input_file(filename):
    """
    讀取並解析輸入檔。
    使用字串切割取代 Regex，徹底避免語法錯誤。
    """
    puzzles = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
    except FileNotFoundError:
        print("Input file not found.")
        return {}

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        
        # 尋找題號
        if '$' in line:
            pid_match = re.search(r'\$(\d+)', line)
            if not pid_match:
                idx += 1
                continue
            
            pid = f"${pid_match.group(1)}"
            idx += 1
            
            raw_hints_buffer = []
            
            # 讀取該題號下的所有內容
            while idx < len(lines):
                curr_line = lines[idx]
                if '$' in curr_line: break
                
                idx += 1
                if not curr_line: continue

                # --- 關鍵修正：使用 split 清除 ---
                # 不使用 Regex，直接用 ']' 切割，取最後一段
                # 範例1: "1 1 2" -> 分割後取最後 -> " 1 1 2" (成功)
                # 範例2: "3 1 1" -> 分割後取最後 -> "3 1 1" (沒變，成功)
                if "source" in curr_line:
                    clean_line = curr_line.split(']')[-1]
                else:
                    clean_line = curr_line
                
                # 解析數字
                # 移除行內註釋 (#之後的內容)
                clean_line = clean_line.split('#')[0]
                numbers = re.findall(r'\d+', clean_line)
                hints = [int(x) for x in numbers]
                
                if not hints: continue

                # 檢查合法性 (過濾掉像 '420' 這種巨大的 source ID)
                if is_valid_hint(hints, 25):
                    raw_hints_buffer.append(hints)

            # --- 雙向去噪邏輯 ---
            # 如果行數超過 50，代表讀到了頁碼雜訊
            while len(raw_hints_buffer) > 50:
                first = raw_hints_buffer[0]
                last = raw_hints_buffer[-1]
                
                # 判斷是否為「疑似頁碼」(單個且很小的數字)
                first_is_noise = (len(first) == 1 and first[0] < 50)
                last_is_noise = (len(last) == 1 and last[0] < 50)
                
                if first_is_noise:
                    print(f"[{pid}] Dropping START noise: {first}")
                    raw_hints_buffer.pop(0)
                elif last_is_noise:
                    print(f"[{pid}] Dropping END noise: {last}")
                    raw_hints_buffer.pop(-1)
                else:
                    # 如果都不是明顯頁碼，優先從後面刪除 (假設多餘的是下一題殘留)
                    print(f"[{pid}] Dropping extra line from END: {last}")
                    raw_hints_buffer.pop(-1)

            if len(raw_hints_buffer) == 50:
                col_hints = raw_hints_buffer[:25]
                row_hints = raw_hints_buffer[25:]
                puzzles[pid] = (col_hints, row_hints)
            else:
                print(f"Error: Puzzle {pid} has {len(raw_hints_buffer)} lines. Skipping.")
        else:
            idx += 1
            
    return puzzles

# ======================================================
# Nonogram Solver Class (保持穩定)
# ======================================================
class NonogramSolver:
    def __init__(self, col_hints, row_hints, pid="Unknown"):
        self.N = 25
        self.col_hints = col_hints
        self.row_hints = row_hints
        self.pid = pid
        self.grid = [[UNKNOWN] * self.N for _ in range(self.N)]
        self.start_time = 0

    def solve(self):
        self.start_time = time.time()
        if not self._propagate(): return None 
        if self._is_solved(): return self.grid
        if self._backtrack(): return self.grid
        return None

    def _is_solved(self):
        for r in range(self.N):
            for c in range(self.N):
                if self.grid[r][c] == UNKNOWN: return False
        return True

    def _propagate(self):
        changed = True
        while changed:
            if time.time() - self.start_time > MAX_TIME_PER_PUZZLE: return True
            changed = False
            
            for r in range(self.N):
                new_line = self._solve_line_complete(self.grid[r], self.row_hints[r])
                if new_line is None: return False
                if new_line != self.grid[r]:
                    self.grid[r] = new_line
                    changed = True
            
            for c in range(self.N):
                current_col = [self.grid[r][c] for r in range(self.N)]
                new_col = self._solve_line_complete(current_col, self.col_hints[c])
                if new_col is None: return False
                if new_col != current_col:
                    for r in range(self.N):
                        self.grid[r][c] = new_col[r]
                    changed = True
        return True

    def _solve_line_complete(self, line, hints):
        N = self.N
        if not hints:
            for x in line:
                if x == FILLED: return None
            return [EMPTY] * N
        min_len = sum(hints) + len(hints) - 1
        if min_len > N: return None

        memo = {}
        can_be_0 = [False] * N
        can_be_1 = [False] * N
        found_any_solution = [False] 

        def recursive_search(hint_idx, start_pos):
            state = (hint_idx, start_pos)
            if state in memo: return memo[state]
            
            if hint_idx == len(hints):
                for k in range(start_pos, N):
                    if line[k] == FILLED:
                        memo[state] = False
                        return False
                found_any_solution[0] = True
                for k in range(start_pos, N): can_be_0[k] = True
                return True

            remaining_len = sum(hints[hint_idx:]) + (len(hints) - 1 - hint_idx)
            if start_pos + remaining_len > N:
                memo[state] = False
                return False

            current_block_len = hints[hint_idx]
            limit = N - remaining_len + 1
            has_valid_path = False
            
            for pos in range(start_pos, limit):
                gap_valid = True
                for k in range(start_pos, pos):
                    if line[k] == FILLED:
                        gap_valid = False
                        break
                if not gap_valid: break 

                block_valid = True
                for k in range(pos, pos + current_block_len):
                    if line[k] == EMPTY:
                        block_valid = False
                        break
                
                if block_valid and (pos + current_block_len < N):
                    if line[pos + current_block_len] == FILLED:
                        block_valid = False
                
                if block_valid:
                    if recursive_search(hint_idx + 1, pos + current_block_len + 1):
                        has_valid_path = True
                        for k in range(start_pos, pos): can_be_0[k] = True
                        for k in range(pos, pos + current_block_len): can_be_1[k] = True
                        if pos + current_block_len < N: can_be_0[pos + current_block_len] = True

            memo[state] = has_valid_path
            return has_valid_path

        recursive_search(0, 0)
        if not found_any_solution[0]: return None

        new_line = list(line)
        for i in range(N):
            if can_be_1[i] and not can_be_0[i]: new_line[i] = FILLED
            elif can_be_0[i] and not can_be_1[i]: new_line[i] = EMPTY
        return new_line

    def _backtrack(self):
        best_r, best_c = -1, -1
        for r in range(self.N):
            for c in range(self.N):
                if self.grid[r][c] == UNKNOWN:
                    best_r, best_c = r, c
                    break
            if best_r != -1: break
        
        if best_r == -1: return True

        saved_grid = deepcopy(self.grid)
        self.grid[best_r][best_c] = FILLED
        if self._propagate():
            if self._backtrack(): return True
        
        self.grid = saved_grid
        self.grid[best_r][best_c] = EMPTY
        if self._propagate():
            if self._backtrack(): return True
            
        return False

# ======================================================
# 主程式
# ======================================================
def main():
    input_file = 'taai2019.txt' 
    output_file = 'result_py2.txt'
    
    print(f"Parsing {input_file}...")
    all_puzzles = parse_input_file(input_file)
    print(f"Loaded {len(all_puzzles)} puzzles.")

    with open(output_file, 'w') as f_out:
        for pid, (col_hints, row_hints) in all_puzzles.items():
            print(f"Solving {pid}...")
            
            solver = NonogramSolver(col_hints, row_hints, pid)
            sol = solver.solve()
            
            f_out.write(f"{pid}\n")
            if sol:
                print(f"-> Solved {pid}!")
                for row in sol:
                    # 使用雙空格
                    f_out.write("  ".join(["1" if x==1 else "0" for x in row]) + "\n")
            else:
                print(f"-> {pid} Unsolvable.")
                f_out.write("Unsolvable\n")
            
            f_out.flush()

if __name__ == "__main__":
    main()