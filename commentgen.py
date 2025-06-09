def wrap_text(text, target_width=40):
    words = text.split()
    lines = []
    current_line = ["#"]

    current_len = 0
    for word in words:
        if current_len + len(word) + len(current_line) > target_width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
        else:
            current_line.append(word)
            current_len += len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return "\n# ".join(lines)

while True:
    print(wrap_text(input("Please input your text: "), target_width=46))