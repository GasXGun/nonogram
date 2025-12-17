import sys
import time
import random

# Constants
UNKNOWN = -1
EMPTY = 0
FILLED = 1


class NonogramSolver:
    def __init__(self, col_hints, row_hints):
        self.N = 25
        self.col_hints = col_hints
        self.row_hints = row_hints
        self.grid = [[UNKNOWN] * self.N for _ in range(self.N)]

    def solve(self):
        
        # 隨機生成解答
        for r in range(self.N):
            for c in range(self.N):
                self.grid[r][c] = random.choice([EMPTY, FILLED])
        return self.grid

    

def parse_line(line):
    return list(map(int, line.strip().split()))

def main():
    # input_file = 'taai2019.txt'
    input_file = 'test_input.txt'
    output_file = 'result_py.txt'
    
    try:
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    with open(output_file, 'w') as f_out:
        idx = 0
        while idx < len(lines):
            if lines[idx].startswith('$'):
                start_time = time.time()
                
                pid = lines[idx]
                idx += 1
                
                col_hints = []
                for _ in range(25):
                    if idx < len(lines):
                        col_hints.append(parse_line(lines[idx]))
                        idx += 1
                
                row_hints = []
                for _ in range(25):
                    if idx < len(lines):
                        row_hints.append(parse_line(lines[idx]))
                        idx += 1
                
                print(f"Solving {pid}...")
                solver = NonogramSolver(col_hints, row_hints)
                solution = solver.solve()
                
                f_out.write(f"{pid}\n")
                if solution:
                    for row in solution:
                        out_row = []
                        for cell in row:
                            out_row.append("1" if cell == FILLED else "0")
                        f_out.write("\t".join(out_row) + "\n")
                else:
                    f_out.write("Unsolvable\n")
                f_out.flush()
                
            else:
                idx += 1

if __name__ == "__main__":
    main()
