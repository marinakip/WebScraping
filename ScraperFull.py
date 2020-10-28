import pendulum

start = pendulum.datetime(2005, 1, 1)
end = pendulum.datetime(2007, 1, 1)
period = pendulum.period(start, end)

for dt in period.range('weeks'):
    print(dt.to_date_string())



#start = '2005-01-01'
#end = '2017-01-01'

#start_url = "https://github.com/search?utf8=%E2%9C%93&q=diabetes+created%3A2005-01-01..2017-01-01&type=Repositories"
start_url = "https://github.com/search?utf8=%E2%9C%93&q=diabetes+created%3A{}..{}&type=Repositories".format(start, end)
