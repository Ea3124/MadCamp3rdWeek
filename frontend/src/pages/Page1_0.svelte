<!-- // src/routes/About.svelte -->
<script lang="ts">
    import { onMount } from 'svelte';
    import { navigate } from 'svelte-routing';
    // 이미지 import
    import sculpture1 from '../assets/images_tab1/statue_1.webp';
    import sculpture2 from '../assets/images_tab1/statue_2.webp';
    import sculpture3 from '../assets/images_tab1/statue_3.webp';
    import sculpture4 from '../assets/images_tab1/statue_4.webp';
    import sculpture5 from '../assets/images_tab1/statue_5.webp';

    // 슬라이더 애니메이션을 위한 변수
    let isHovered = false;

    let resultImage: string | null = null;

    // 컴포넌트가 제거될 때 URL 정리
    import { onDestroy } from 'svelte';
    onDestroy(() => {
        if (resultImage) {
            URL.revokeObjectURL(resultImage);
        }
    });

    onMount(() => {
        window.scrollTo(0, 0);  // 페이지 로드 시 스크롤을 맨 위로
    });

    function handleNextPage() {
        navigate('/theme1_1');
    }
</script>

<main>
    <div class="section-divider"></div>
    
    <section class="description-section">
        <!-- 상단 슬라이더 -->
        <div class="slider-container-top" 
            role="region"
            aria-label="상단 이미지 슬라이더"
            on:mouseenter={() => isHovered = true}
            on:mouseleave={() => isHovered = false}>
            <div class="slider" >
                <!-- 첫 번째 세트 -->
                <img src={sculpture1} alt="sculpture 1">
                <img src={sculpture2} alt="sculpture 2">
                <img src={sculpture3} alt="sculpture 3">
                <img src={sculpture4} alt="sculpture 4">
                <img src={sculpture5} alt="sculpture 5">
                <!-- 순환을 위한 복제 세트 -->
                <img src={sculpture1} alt="sculpture 1">
                <img src={sculpture2} alt="sculpture 2">
                <img src={sculpture3} alt="sculpture 3">
                <img src={sculpture4} alt="sculpture 4">
                <img src={sculpture5} alt="sculpture 5">
                <!-- 첫 번째 세트 -->
                <img src={sculpture1} alt="sculpture 1">
                <img src={sculpture2} alt="sculpture 2">
                <img src={sculpture3} alt="sculpture 3">
                <img src={sculpture4} alt="sculpture 4">
                <img src={sculpture5} alt="sculpture 5">
                <!-- 순환을 위한 복제 세트 -->
                <img src={sculpture1} alt="sculpture 1">
                <img src={sculpture2} alt="sculpture 2">
                <img src={sculpture3} alt="sculpture 3">
                <img src={sculpture4} alt="sculpture 4">
                <img src={sculpture5} alt="sculpture 5">
            </div>
        </div>
        
        <h1 class="title" on:click={handleNextPage}>1. 조각상 생성 »</h1>

        <!-- 하단 슬라이더 -->
        <div class="slider-container-bottom" 
            role="region"
            aria-label="하단 이미지 슬라이더"
            on:mouseenter={() => isHovered = true}
            on:mouseleave={() => isHovered = false}>
            <div class="slider">
                <!-- 첫 번째 세트 -->
                <img src={sculpture1} alt="sculpture 1">
                <img src={sculpture2} alt="sculpture 2">
                <img src={sculpture3} alt="sculpture 3">
                <img src={sculpture4} alt="sculpture 4">
                <img src={sculpture5} alt="sculpture 5">
                <!-- 순환을 위한 복제 세트 -->
                <img src={sculpture1} alt="sculpture 1">
                <img src={sculpture2} alt="sculpture 2">
                <img src={sculpture3} alt="sculpture 3">
                <img src={sculpture4} alt="sculpture 4">
                <img src={sculpture5} alt="sculpture 5">
                <!-- 첫 번째 세트 -->
                <img src={sculpture1} alt="sculpture 1">
                <img src={sculpture2} alt="sculpture 2">
                <img src={sculpture3} alt="sculpture 3">
                <img src={sculpture4} alt="sculpture 4">
                <img src={sculpture5} alt="sculpture 5">
                <!-- 순환을 위한 복제 세트 -->
                <img src={sculpture1} alt="sculpture 1">
                <img src={sculpture2} alt="sculpture 2">
                <img src={sculpture3} alt="sculpture 3">
                <img src={sculpture4} alt="sculpture 4">
                <img src={sculpture5} alt="sculpture 5">
            </div>
        </div>
    </section>

    <div class="footer">
        <div class="footer-content">
            내가 좋아하고, 내가 원하는 사물, 동물, 혹은 그 어떤 것이든,
            원하는 것을 생성하세요. 
        </div>
    </div>

</main>



<style>
    :global(body) {
        background: #111111;
        margin: 0;
        padding: 0;
        overflow-x: hidden;
    }

    main {
        padding: 0;
    }

    .description-section {
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        padding: 0;
        margin: 0;
        overflow: hidden;
    }

    .description-section h1 {
        font-size: 120px;
        font-weight: bold;
        margin-bottom: 80px;
        margin-top: 0px;
        color: #000000;
        padding: 30px 60px;
        white-space: nowrap;
        position: relative;
        z-index: 10;
        text-shadow: none;
        letter-spacing: 5px;
        transition: transform 0.3s ease;
        cursor: pointer; /* 기본 커서를 버튼 스타일로 변경 */
    }

    /* 형광펜 밑줄 효과 */
    .description-section h1::after {
        content: '';
        position: absolute;
        bottom: 40px; /* 텍스트에서 밑줄의 거리 */
        left: 0;
        width: 100%; /* 밑줄이 텍스트 너비를 덮도록 설정 */
        height: 40px; /* 형광펜 두께 */
        background-color: aquamarine; /* 형광펜 색상 (노란색) */
        z-index: -1; /* 텍스트 아래로 이동 */
        opacity: 0.7; /* 형광펜 투명도 */
        transform: skewX(-20deg); /* 약간의 기울임 효과 */
    }

    .description-section h1:hover {
        transform: scale(1.02) translateY(-5px);
        cursor: pointer; /* 기본 커서를 버튼 스타일로 변경 */
    }

    .section-divider {
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            #CCCCCC 50%, 
            transparent 100%
        );
        margin: 0;
        opacity: 0.5;
    }
    .slider-container-top {
        position: absolute;
        top: -10px;
        z-index: 1;
    }

    .slider-container-bottom {
        position: absolute;
        bottom: -10px;
        z-index: 1;
    }

    .slider-container-top .slider {
        animation: scroll 20s linear infinite reverse;
    }

    .slider-container-bottom .slider {
        animation: scroll 20s linear infinite;
    }

    .slider {
        display: flex;
        gap: 20px;
        animation: scroll 20s linear infinite reverse;
    }

    @keyframes scroll {
        0% {
            transform: translateX(calc(-300px * 5 - 20px * 5));
        }
        100% {
            transform: translateX(0);
        }
    }

    .slider img {
        width: 240px;
        height: 320px;
        object-fit: cover;
        border-radius: 10px;
        flex-shrink: 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .slider img:hover {
        transform: translateY(-10px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .footer {
        position: fixed;
        bottom: 30%;
        left: 0;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px;
        z-index: 100;
    }

    .footer-content {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(255, 255, 255, 0.1);
        padding: 12px 24px;
        border-radius: 12px;
        backdrop-filter: blur(5px);
        font-size: 20px;
    }
</style>

