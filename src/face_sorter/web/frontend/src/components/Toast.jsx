import React, { useState, useEffect } from 'react';
import './Toast.css';
import toastService from '../services/toast';

const getIcon = (type) => {
  const icons = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ',
  };
  return icons[type] || icons.info;
};

function Toast() {
  const [toasts, setToasts] = useState([]);

  useEffect(() => {
    const unsubscribe = toastService.subscribe((newToasts) => {
      setToasts([...newToasts]);
    });
    return () => {
      unsubscribe();
    };
  }, []);

  const handleClose = (e, toast) => {
    e.stopPropagation();
    toastService.remove(toast.id);
  };

  const handleClick = (toast) => {
    console.log('Toast clicked:', toast);
  };

  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`toast-item toast-${toast.type}`}
          onClick={() => handleClick(toast)}
        >
          <div className="toast-icon">
            {getIcon(toast.type)}
          </div>
          <div className="toast-content">
            <p className="toast-message">{toast.message}</p>
          </div>
          <button className="toast-close" onClick={(e) => handleClose(e, toast)}>
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}

export default Toast;