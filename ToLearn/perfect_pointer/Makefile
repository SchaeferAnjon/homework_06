# 简单的Makefile
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
TARGET = main
SOURCES = main.cpp
HEADERS = student.hpp utils.hpp

# 默认目标
all: $(TARGET)

# 编译目标
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

# 运行程序
run: $(TARGET)
	./$(TARGET)

# 清理生成的文件
clean:
	rm -f $(TARGET)

# 重新编译
rebuild: clean all

# 调试版本
debug: CXXFLAGS += -DDEBUG -O0
debug: $(TARGET)

# 发布版本
release: CXXFLAGS += -O2 -DNDEBUG
release: $(TARGET)

# 显示帮助
help:
	@echo "可用的目标:"
	@echo "  all      - 编译程序 (默认)"
	@echo "  run      - 编译并运行程序"
	@echo "  clean    - 清理生成的文件"
	@echo "  rebuild  - 重新编译"
	@echo "  debug    - 编译调试版本"
	@echo "  release  - 编译发布版本"
	@echo "  help     - 显示此帮助信息"

.PHONY: all run clean rebuild debug release help 