import requests
import threading

def test_query_endpoint():
    url = "http://127.0.0.1:8000/query"
    payload = {"question": "法院判决河南省南阳市人民政府需要赔偿征地补偿款多少钱"}
    response = requests.post(url, json=payload)
    assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"
    data = response.json()
    assert "841.1545万元" in data.get("answer", ""), f"Answer does not contain expected amount. Answer: {data.get('answer', '')}"
    print("赔偿金额问题测试通过。")

def test_query_court():
    url = "http://127.0.0.1:8000/query"
    payload = {"question": "河南省高级人民法院审理的南阳某某房地产开发有限公司状告河南省南阳市人民政府征地补偿款纠纷案的判决号是多少"}
    response = requests.post(url, json=payload)
    assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"
    data = response.json()
    print("法院审理问题返回：", data.get("answer", "(2020)豫行终3143号"))

if __name__ == "__main__":
    threads = []
    for func in [test_query_endpoint, test_query_court]:
        t = threading.Thread(target=func)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("All threaded tests finished.")
