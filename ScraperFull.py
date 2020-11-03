import pendulum

def create_start_url():
    query = 'diabetes'
    start = pendulum.datetime(2005, 1, 1)
    end = pendulum.datetime(2007, 1, 6)
    period = pendulum.period(start, end)

    weeks_list = []
    for dt in period.range('weeks'):
        date = dt.to_date_string()
        weeks_list.append(date)
        print(date)

    for date in range(len(weeks_list)):
        start = weeks_list[::2]
        end = weeks_list[1::2]
        start = str(start)
        end = str(end)
        start_url = "https://github.com/search?utf8=%E2%9C%93&q={}+created%3A{}..{}&type=Repositories".format(query, start,
                                                                                                            end)
        return start_url
#print(len(start))
#print(len(end))


#start = '2005-01-01'
#end = '2017-01-01'

#start_url = "https://github.com/search?utf8=%E2%9C%93&q=diabetes+created%3A2005-01-01..2017-01-01&type=Repositories"
