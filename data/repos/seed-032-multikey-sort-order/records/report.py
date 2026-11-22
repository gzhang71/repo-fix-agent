from .sorting import sort_by


def top_earners_by_dept(rows):
    """Lines like 'eng  Ana  120000', grouped by department, highest salary first."""
    return [f"{r['dept']:<5}{r['name']:<6}{r['salary']}" for r in sort_by(rows, "dept", "-salary")]
