// 예시: src/lib/api.ts
export async function fetchHello() {
    const response = await fetch('http://0.0.0.0:8000/api/hello');
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  }
  