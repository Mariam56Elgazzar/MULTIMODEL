"""
Rate Limiter V2 - محسّن مع حماية ذكية من تجاوز الحدود
✅ إدارة يومية وشهرية للـ tokens
✅ تنبيهات قبل تجاوز الحدود
✅ استراتيجية تقليل ديناميكية
"""

import time
import asyncio
from typing import Optional, Dict, Any
from collections import deque
from datetime import datetime, timedelta
import logging
import json
import os

logger = logging.getLogger(__name__)


class AdvancedRateLimiter:
    """
    Rate Limiter متقدم مع إدارة ذكية للموارد
    """
    
    def __init__(
        self,
        requests_per_minute: int = 30,
        tokens_per_minute: int = 14400,
        tokens_per_day: int = 600_000,
        tokens_per_month: int = 15_000_000,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        use_exponential_backoff: bool = True,
        safety_margin_percent: float = 10.0,
        config_file: str = "rate_limiter_state.json"
    ):
        # حدود الـ Groq API
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_day = tokens_per_day
        self.tokens_per_month = tokens_per_month
        
        # إعادة المحاولة
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.use_exponential_backoff = use_exponential_backoff
        
        # هامش الأمان (لتجنب التجاوز الأخير)
        self.safety_margin_percent = safety_margin_percent
        self.safety_margin_day = int(tokens_per_day * safety_margin_percent / 100)
        self.safety_margin_month = int(tokens_per_month * safety_margin_percent / 100)
        
        # تتبع الاستخدام
        self.request_times = deque(maxlen=requests_per_minute)
        self.token_usage_minute = deque()  # (timestamp, tokens)
        self.token_usage_day = deque()
        self.token_usage_month = deque()
        
        # إحصائيات
        self.total_requests = 0
        self.total_tokens = 0
        self.failed_requests = 0
        self.total_wait_time = 0.0
        
        self.config_file = config_file
        self.load_state()
        
        logger.info(f"""
        ✅ RateLimiter V2 initialized:
        - {requests_per_minute} requests/min
        - {tokens_per_minute:,} tokens/min
        - {tokens_per_day:,} tokens/day
        - {tokens_per_month:,} tokens/month
        - Safety margin: {safety_margin_percent}%
        """)
    
    def load_state(self):
        """تحميل الحالة المحفوظة"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.total_tokens = data.get('total_tokens', 0)
                    self.total_requests = data.get('total_requests', 0)
                    logger.info(f"Loaded state: {self.total_tokens} tokens, {self.total_requests} requests")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def save_state(self):
        """حفظ الحالة الحالية"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    'total_tokens': self.total_tokens,
                    'total_requests': self.total_requests,
                    'timestamp': datetime.now().isoformat(),
                    'failed_requests': self.failed_requests
                }, f)
        except Exception as e:
            logger.warning(f"Could not save state: {e}")
    
    def _clean_old_entries(self):
        """تنظيف الإدخالات القديمة"""
        current_time = datetime.now()
        
        # تنظيف دقيقة واحدة
        cutoff_minute = current_time - timedelta(minutes=1)
        while self.token_usage_minute and self.token_usage_minute[0][0] < cutoff_minute:
            self.token_usage_minute.popleft()
        
        # تنظيف يوم واحد
        cutoff_day = current_time - timedelta(days=1)
        while self.token_usage_day and self.token_usage_day[0][0] < cutoff_day:
            self.token_usage_day.popleft()
        
        # تنظيف شهر واحد
        cutoff_month = current_time - timedelta(days=30)
        while self.token_usage_month and self.token_usage_month[0][0] < cutoff_month:
            self.token_usage_month.popleft()
    
    def _get_current_usage(self) -> Dict[str, int]:
        """الحصول على الاستخدام الحالي"""
        self._clean_old_entries()
        
        current_requests = len(self.request_times)
        tokens_minute = sum(t for _, t in self.token_usage_minute)
        tokens_day = sum(t for _, t in self.token_usage_day)
        tokens_month = sum(t for _, t in self.token_usage_month)
        
        return {
            'requests': current_requests,
            'tokens_minute': tokens_minute,
            'tokens_day': tokens_day,
            'tokens_month': tokens_month
        }
    
    def _get_wait_time_and_reason(self, estimated_tokens: int = 100) -> tuple[float, Optional[str]]:
        """حساب وقت الانتظار والسبب"""
        usage = self._get_current_usage()
        wait_time = 0.0
        reason = None
        
        # فحص حد الطلبات في الدقيقة
        if usage['requests'] >= self.requests_per_minute:
            if self.request_times:
                time_until_oldest_expires = 60 - (datetime.now() - self.request_times[0]).total_seconds()
                if time_until_oldest_expires > wait_time:
                    wait_time = time_until_oldest_expires
                    reason = f"Request rate limit ({usage['requests']}/{self.requests_per_minute})"
        
        # فحص حد الـ tokens في الدقيقة
        if usage['tokens_minute'] + estimated_tokens > self.tokens_per_minute:
            if self.token_usage_minute:
                time_until_oldest = 60 - (datetime.now() - self.token_usage_minute[0][0]).total_seconds()
                if time_until_oldest > wait_time:
                    wait_time = time_until_oldest
                    reason = f"Token rate limit ({usage['tokens_minute']}/{self.tokens_per_minute})"
        
        # فحص حد الـ tokens اليومي (مع هامش أمان)
        if usage['tokens_day'] + estimated_tokens > (self.tokens_per_day - self.safety_margin_day):
            remaining_day = self.tokens_per_day - usage['tokens_day'] - self.safety_margin_day
            if remaining_day <= 0:
                wait_time = float('inf')
                reason = f"⚠️ Daily token limit reached! ({usage['tokens_day']}/{self.tokens_per_day})"
            elif remaining_day < estimated_tokens:
                logger.warning(f"⚠️ Daily token limit approaching! Remaining: {remaining_day}")
        
        # فحص حد الـ tokens الشهري
        if usage['tokens_month'] + estimated_tokens > (self.tokens_per_month - self.safety_margin_month):
            remaining_month = self.tokens_per_month - usage['tokens_month'] - self.safety_margin_month
            if remaining_month <= 0:
                wait_time = float('inf')
                reason = f"❌ Monthly token limit exceeded! ({usage['tokens_month']}/{self.tokens_per_month})"
        
        return wait_time, reason
    
    async def acquire(self, estimated_tokens: int = 100) -> Dict[str, Any]:
        """
        حجز موارد للطلب
        """
        wait_time, reason = self._get_wait_time_and_reason(estimated_tokens)
        
        if wait_time == float('inf'):
            logger.error(f"🛑 {reason}")
            return {
                'allowed': False,
                'reason': reason,
                'wait_time': -1
            }
        
        if wait_time > 0:
            logger.warning(f"⏳ {reason} - Waiting {wait_time:.2f}s")
            self.total_wait_time += wait_time
            await asyncio.sleep(wait_time)
        
        # تسجيل الطلب
        current_time = datetime.now()
        self.request_times.append(current_time)
        self.token_usage_minute.append((current_time, estimated_tokens))
        self.token_usage_day.append((current_time, estimated_tokens))
        self.token_usage_month.append((current_time, estimated_tokens))
        self.total_requests += 1
        
        return {
            'allowed': True,
            'reason': None,
            'wait_time': wait_time
        }
    
    def release(self, actual_tokens: int):
        """تحديث استخدام الـ tokens الفعلي"""
        self.total_tokens += actual_tokens
        
        # حفظ الحالة بعد كل طلب
        self.save_state()
        
        # تنبيهات
        usage = self._get_current_usage()
        
        daily_percent = (usage['tokens_day'] / self.tokens_per_day) * 100
        if daily_percent > 80:
            logger.warning(f"⚠️ Daily limit {daily_percent:.1f}% used")
        
        monthly_percent = (usage['tokens_month'] / self.tokens_per_month) * 100
        if monthly_percent > 80:
            logger.warning(f"🔴 Monthly limit {monthly_percent:.1f}% used")
    
    def can_make_request(self, estimated_tokens: int) -> bool:
        """التحقق السريع من الإمكانية"""
        wait_time, reason = self._get_wait_time_and_reason(estimated_tokens)
        return wait_time != float('inf')
    
    def get_statistics(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات"""
        usage = self._get_current_usage()
        
        return {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'failed_requests': self.failed_requests,
            'total_wait_time': f"{self.total_wait_time:.2f}s",
            'current': {
                'requests_per_minute': f"{usage['requests']}/{self.requests_per_minute}",
                'tokens_per_minute': f"{usage['tokens_minute']:,}/{self.tokens_per_minute:,}",
                'tokens_per_day': f"{usage['tokens_day']:,}/{self.tokens_per_day:,} ({(usage['tokens_day']/self.tokens_per_day*100):.1f}%)",
                'tokens_per_month': f"{usage['tokens_month']:,}/{self.tokens_per_month:,} ({(usage['tokens_month']/self.tokens_per_month*100):.1f}%)"
            },
            'safety_margins': {
                'daily': f"{self.safety_margin_day:,} tokens",
                'monthly': f"{self.safety_margin_month:,} tokens"
            }
        }
    
    def print_statistics(self):
        """طباعة الإحصائيات"""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("📊 RATE LIMITER STATISTICS")
        print("="*60)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Total Tokens Used: {stats['total_tokens']:,}")
        print(f"Failed Requests: {stats['failed_requests']}")
        print(f"Total Wait Time: {stats['total_wait_time']}")
        print("\n📈 Current Usage:")
        for key, value in stats['current'].items():
            print(f"  {key}: {value}")
        print("\n🛡️ Safety Margins:")
        for key, value in stats['safety_margins'].items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")


# Singleton instance
_global_limiter: Optional[AdvancedRateLimiter] = None


def get_rate_limiter(config: Dict[str, Any] = None) -> AdvancedRateLimiter:
    """الحصول على instance من Rate Limiter"""
    global _global_limiter
    
    if _global_limiter is None:
        if config is None:
            config = {
                'requests_per_minute': 30,
                'tokens_per_minute': 14400,
                'tokens_per_day': 600_000,
                'tokens_per_month': 15_000_000,
                'retry_attempts': 3,
                'retry_delay': 2.0,
                'safety_margin_percent': 10.0
            }
        
        _global_limiter = AdvancedRateLimiter(**config)
    
    return _global_limiter


def reset_rate_limiter():
    """إعادة تعيين Rate Limiter"""
    global _global_limiter
    _global_limiter = None