
def take_part_of_file(infile, outfile, skip):
    with open(infile, encoding="utf8", errors='ignore') as f:
        lines = f.readlines()
        desired_lines = lines[1::skip]

    with open(outfile, 'w') as f:
        for l in desired_lines:
            f.write("%s" % l)