"""
กำหนด Custom Exceptions สำหรับโปรเจกต์
"""

class BoardNotFoundException(Exception):
    """
    Exception ที่จะถูกโยน (raise) เมื่อไม่สามารถค้นหากระดานหมากรุกในภาพได้
    """
    pass

class LinesNotFoundException(Exception):
    """
    Exception ที่จะถูกโยน เมื่อไม่พบเส้นเพียงพอในภาพ
    """
    pass