<script lang="ts">
    import { onMount } from 'svelte';
    import { writable } from 'svelte/store';
  
    // 타입 정의
    interface GenerateRequest {
      prompt: string;
      num_images: number;
    }
  
    interface GenerateResponse {
      images: string[]; // Base64 인코딩된 이미지 배열
    }
  
    // 상태 관리
    let prompt: string = '';
    let numImages: number = 1;
    const images = writable<string[]>([]);
    const error = writable<string | null>(null);
    const isLoading = writable<boolean>(false);
  
    // 이미지 생성 함수
    async function generateImage(request: GenerateRequest): Promise<string[]> {
      isLoading.set(true);
      error.set(null);
      try {
        const response = await fetch('http://127.0.0.1:8001/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(request)
        });
  
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || '이미지 생성 실패');
        }
  
        const data: GenerateResponse = await response.json();
        return data.images;
      } catch (err) {
        if (err instanceof Error) {
          error.set(err.message);
        } else {
          error.set('알 수 없는 오류가 발생했습니다.');
        }
        return [];
      } finally {
        isLoading.set(false);
      }
    }
  
    // 폼 제출 핸들러
    async function handleSubmit(event: Event) {
      event.preventDefault();
      const request: GenerateRequest = {
        prompt,
        num_images: numImages
      };
      const generatedImages = await generateImage(request);
      images.set(generatedImages);
    }
  </script>
  
  <style>
    .container {
      max-width: 600px;
      margin: 0 auto;
      padding: 2rem;
      text-align: center;
    }
  
    .input-group {
      margin-bottom: 1rem;
    }
  
    label {
      display: block;
      margin-bottom: 0.5rem;
      font-weight: bold;
    }
  
    input, select, button {
      width: 100%;
      padding: 0.5rem;
      font-size: 1rem;
    }
  
    .images {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      justify-content: center;
      margin-top: 2rem;
    }
  
    .images img {
      max-width: 200px;
      border: 1px solid #ccc;
      border-radius: 8px;
    }
  
    .error {
      color: red;
      margin-top: 1rem;
    }
  
    .loading {
      margin-top: 1rem;
      font-style: italic;
    }
  </style>
  
  <div class="container">
    <h1>이미지 생성기</h1>
    <form on:submit|preventDefault={handleSubmit}>
      <div class="input-group">
        <label for="prompt">프롬프트</label>
        <input
          id="prompt"
          type="text"
          bind:value={prompt}
          placeholder="예: vintage monkey illustration"
          required
        />
      </div>
      <div class="input-group">
        <label for="numImages">생성할 이미지 수</label>
        <select id="numImages" bind:value={numImages}>
          {#each Array.from({ length: 10 }, (_, i) => i + 1) as number}
            <option value={number}>{number}</option>
          {/each}
        </select>
      </div>
      <button type="submit" disabled={$isLoading}>이미지 생성</button>
    </form>
  
    {#if $isLoading}
      <div class="loading">이미지 생성 중...</div>
    {/if}
  
    {#if $error}
      <div class="error">오류: {$error}</div>
    {/if}
  
    <div class="images">
      {#each $images as imgBase64, index}
        <img src={`data:image/png;base64,${imgBase64}`} alt={`Generated Image ${index + 1}`} />
      {/each}
    </div>
  </div>
  