def convert(form_values):

    converted = []
    converted.append((float(form_values[0]) - (75.208882)) / 9.865026)
    converted.append((float(form_values[1]) - (1.203597)) / 0.135091)
    converted.append((float(form_values[2]) - (0.737130)) / 0.042670)
    converted.append((float(form_values[3]) - (1477.062500)) / 170.653795)
    gender = form_values[4]

    if (gender == 'Male'):

        converted.append(1)

    else:

        converted.append(0)

    educ = form_values[5]

    if (educ == '1'):

        converted.append(0)
        converted.append(0)
        converted.append(0)
        converted.append(0)

    elif (educ == '2'):

        converted.append(1)
        converted.append(0)
        converted.append(0)
        converted.append(0)

    elif (educ == '3'):

        converted.append(0)
        converted.append(1)
        converted.append(0)
        converted.append(0)

    elif (educ == '4'):

        converted.append(0)
        converted.append(0)
        converted.append(1)
        converted.append(0)

    elif (educ == '5'):

        converted.append(0)
        converted.append(0)
        converted.append(0)
        converted.append(1)

    ses = form_values[6]

    if (ses == '1'):

        converted.append(0)
        converted.append(0)
        converted.append(0)
        converted.append(0)

    elif (ses == '2'):

        converted.append(1)
        converted.append(0)
        converted.append(0)
        converted.append(0)

    elif (ses == '3'):

        converted.append(0)
        converted.append(1)
        converted.append(0)
        converted.append(0)

    elif (ses == '4'):

        converted.append(0)
        converted.append(0)
        converted.append(1)
        converted.append(0)

    elif (ses == '5'):

        converted.append(0)
        converted.append(0)
        converted.append(0)
        converted.append(1)

    mmse = float(form_values[7])

    if (mmse > 25):

        converted.append(1)
    else:

        converted.append(0)

    return converted