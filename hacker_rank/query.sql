select Employee_ID as eid, Name as name from Employee emp inner join (select Gender, Employee_ID as emp_id, Pay_Scale as ps, Time_since_promotion as tsp from Service where ps > 4.0 AND tsp > 1 AND Gender = 'F' from Service group by emp_id) sal on (emp.eid = sal.emp_no AND emp.Unit = 'Sales');