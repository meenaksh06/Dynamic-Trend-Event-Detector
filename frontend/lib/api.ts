const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function fetchStats() {
  const res = await fetch(`${API_BASE_URL}/api/stats`);
  if (!res.ok) throw new Error('Failed to fetch stats');
  return res.json();
}

export async function fetchTopics() {
  const res = await fetch(`${API_BASE_URL}/api/topics`);
  if (!res.ok) throw new Error('Failed to fetch topics');
  return res.json();
}

export async function fetchTrends() {
  const res = await fetch(`${API_BASE_URL}/api/trends`);
  if (!res.ok) throw new Error('Failed to fetch trends');
  return res.json();
}

export async function fetchForecast(topicId: number) {
  const res = await fetch(`${API_BASE_URL}/api/forecast/${topicId}`);
  if (!res.ok) throw new Error('Failed to fetch forecast');
  return res.json();
}

export async function fetchHybridForecast(topicId: number) {
  const res = await fetch(`${API_BASE_URL}/api/hybrid-forecast/${topicId}`);
  if (!res.ok) throw new Error('Failed to fetch hybrid forecast');
  return res.json();
}

export async function fetchAblation() {
  const res = await fetch(`${API_BASE_URL}/api/ablation`);
  if (!res.ok) throw new Error('Failed to fetch ablation data');
  return res.json();
}

export async function fetchSentimentTimeline() {
  const res = await fetch(`${API_BASE_URL}/api/sentiment-timeline`);
  if (!res.ok) throw new Error('Failed to fetch sentiment timeline');
  return res.json();
}

export async function fetchArticles(topicId: number, limit: number = 10) {
  const res = await fetch(`${API_BASE_URL}/api/articles/${topicId}?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch articles');
  return res.json();
}
